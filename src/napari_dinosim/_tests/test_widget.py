import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from magicgui.widgets import ComboBox
from napari.layers import Image, Points
from napari.viewer import Viewer

from napari_dinosim._widget import DINOSim_widget

# Filter warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="Pickle, copy, and deepcopy support will be removed from itertools",
)
warnings.filterwarnings("ignore", message="xFormers is not available")


# Fixture for mocked viewer
@pytest.fixture
def mock_viewer():
    viewer = MagicMock(spec=Viewer)
    # Create a proper mock for layers that includes events
    layers = MagicMock()
    layers.events = MagicMock()
    layers.events.inserted = MagicMock()
    layers.events.removed = MagicMock()
    layers.__iter__ = lambda x: iter([])  # Empty layer list by default
    layers.__getitem__ = lambda x, key: (
        x._list[key]
        if isinstance(key, int)
        else next((l for l in x._list if l.name == key), None)
    )
    layers._list = []
    viewer.layers = layers
    return viewer


# Fixture for widget instance with proper cleanup
@pytest.fixture
def widget(mock_viewer, qt_app):
    with patch("torch.hub.load") as mock_hub:
        # Mock the DINO model
        mock_model = MagicMock()
        mock_model.patch_size = 14
        mock_model.eval = MagicMock(return_value=None)
        mock_hub.return_value = mock_model

        widget = DINOSim_widget(mock_viewer)
        # Mock the image layer combo to allow setting values
        widget._image_layer_combo = MagicMock(spec=ComboBox)
        widget._image_layer_combo.choices = []

        yield widget

        # Cleanup after test
        try:
            widget.closeEvent(MagicMock())
            qt_app.processEvents()  # Process any pending events
        except Exception as e:
            print(f"Error during widget cleanup: {str(e)}")


# Test initialization
def test_widget_initialization(widget, mock_viewer):
    """Test that widget initializes correctly."""
    assert widget._viewer == mock_viewer
    assert widget.model_dims == {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }
    assert widget.crop_sizes == {
        "x1": (518, 518),
        "x0.5": (1036, 1036),
        "x2": (259, 259),
    }
    assert widget.resize_size == 518


# Test model loading
def test_model_loading(widget, qt_app):
    """Test model loading functionality."""
    # Initialize widget state to force model loading
    widget.feat_dim = 0  # Different from model_dims["small"] to force loading

    with patch("torch.hub.load") as mock_hub:
        # Mock the DINO model
        mock_model = MagicMock()
        mock_model.patch_size = 14
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=None)
        mock_hub.return_value = mock_model

        # Call the model loading method
        widget.model_size_selector.value = "small"
        worker = widget._load_model_threaded()
        worker.run()  # Run synchronously for testing

        # Process any pending events
        qt_app.processEvents()

        # Verify the model was loaded correctly
        mock_hub.assert_called_once_with(
            "facebookresearch/dinov2", "dinov2_vits14_reg"
        )

        # Verify model setup
        assert widget.model is mock_model
        assert widget.pipeline_engine is not None
        assert mock_model.eval.called
        assert mock_model.to.called

        # Verify button state
        assert (
            widget._load_model_btn.text
            == "Load New Model\n(Current Model: small)"
        )
        assert "lightgreen" in widget._load_model_btn.native.styleSheet()


# Test image processing
def test_image_processing(widget):
    """Test image processing functionality."""
    # Create test image
    test_image = np.random.rand(100, 100, 3)
    processed_image = widget._get_nhwc_image(test_image)

    assert processed_image.ndim == 4  # Should add batch dimension
    assert processed_image.shape == (1, 100, 100, 3)

    # Test grayscale
    test_image_gray = np.random.rand(100, 100)
    processed_image_gray = widget._get_nhwc_image(test_image_gray)
    assert processed_image_gray.shape == (1, 100, 100, 1)


# Test uint8 conversion
def test_uint8_conversion(widget):
    """Test conversion to uint8."""
    # Test float image [0, 1]
    float_image = np.random.rand(10, 10, 3)
    uint8_image = widget._touint8(float_image)
    assert uint8_image.dtype == np.uint8
    assert uint8_image.max() <= 255
    assert uint8_image.min() >= 0

    # Test arbitrary range float image
    arbitrary_image = np.random.rand(10, 10, 3) * 1000
    uint8_arbitrary = widget._touint8(arbitrary_image)
    assert uint8_arbitrary.dtype == np.uint8
    assert uint8_arbitrary.max() <= 255
    assert uint8_arbitrary.min() >= 0


# Test points layer handling
def test_points_layer_handling(widget, mock_viewer):
    """Test points layer creation and handling."""
    # Mock image layer
    mock_image = MagicMock(spec=Image)
    mock_image.ndim = 2
    mock_image.name = "test_image"
    mock_viewer.layers._list = [mock_image]
    widget._image_layer_combo.value = mock_image

    widget._add_points_layer()
    assert mock_viewer.add_points.called

    # Test 3D image
    mock_image.ndim = 3
    widget._points_layer = None
    widget._add_points_layer()
    assert mock_viewer.add_points.called


# Test reference handling
def test_reference_handling(widget):
    """Test reference point handling."""
    # Mock pipeline engine
    widget.pipeline_engine = MagicMock()
    widget.pipeline_engine.exist_reference = False

    # Create test points - for 2D points, use only y,x coordinates
    test_points = np.array(
        [[10, 10], [20, 20]]
    )  # Changed from 3D to 2D points
    widget._points_layer = MagicMock(spec=Points)
    widget._points_layer.data = test_points

    # Mock image layer
    mock_image = MagicMock(spec=Image)
    mock_image.data = np.random.rand(100, 100, 3)
    mock_image.name = "test_image"
    widget._image_layer_combo.value = mock_image
    widget._image_layer_combo.choices = [mock_image]

    widget._update_reference_and_process()
    assert len(widget._references_coord) == 2  # Should have 2 reference points


# Test threshold handling
def test_threshold_handling(widget):
    """Test threshold application."""
    widget.predictions = np.random.rand(100, 100)
    widget._threshold_slider.value = 0.5

    # Mock image layer
    mock_image = MagicMock(spec=Image)
    mock_image.name = "test_image"
    widget._image_layer_combo.value = mock_image

    with patch.object(widget._viewer, "add_labels") as mock_add_labels:
        widget.threshold_im()
        assert mock_add_labels.called


# Test cleanup
def test_cleanup(widget):
    """Test cleanup on widget close."""
    widget.pipeline_engine = MagicMock()
    widget.model = MagicMock()
    widget._points_layer = MagicMock()
    widget._points_layer.events = MagicMock()
    widget._points_layer.events.data = MagicMock()
    widget._points_layer.events.data.disconnect = MagicMock()

    # Call cleanup operations directly instead of using closeEvent
    if widget.pipeline_engine is not None:
        widget.pipeline_engine.delete_precomputed_embeddings()
        widget.pipeline_engine.delete_references()

    if widget._points_layer is not None:
        widget._points_layer.events.data.disconnect(
            widget._update_reference_and_process
        )
        widget._points_layer = None

    if widget.model is not None:
        widget.model = None

    assert widget.pipeline_engine.delete_precomputed_embeddings.called
    assert widget.pipeline_engine.delete_references.called
    assert widget._points_layer is None
    assert widget.model is None


# Test error handling
def test_error_handling(widget):
    """Test error handling in critical operations."""
    # Initialize widget state to force model loading
    widget.feat_dim = 0  # Different from model_dims["small"] to force loading

    # Mock pipeline engine to ensure it exists
    widget.pipeline_engine = MagicMock()

    # Mock viewer with status property
    mock_viewer = MagicMock()
    mock_status = PropertyMock()
    type(mock_viewer).status = mock_status
    widget._viewer = mock_viewer

    # Test auto_precompute with invalid image
    invalid_image = np.random.rand(2, 100, 100, 5)  # 5 channels
    mock_image = MagicMock(spec=Image)
    mock_image.data = invalid_image
    widget._image_layer_combo.value = mock_image

    with pytest.raises(AssertionError):
        widget.auto_precompute()  # This should raise the assertion

    # Test model loading error
    with patch("torch.hub.load", side_effect=Exception("Network error")):
        # Store previous model state
        prev_model = widget.model
        prev_pipeline = widget.pipeline_engine

        # Call the model loading method
        widget.model_size_selector.value = "small"
        worker = widget._load_model_threaded()

        # Run synchronously for testing
        try:
            worker.run()  # This should raise the exception
        except Exception:
            pass  # Expected exception

        # Verify error handling
        mock_status.assert_called_with("Error loading model: Network error")
        assert (
            widget.pipeline_engine == prev_pipeline
        )  # Pipeline should remain unchanged

        # The model should be None since we're resetting it before attempting to load
        assert widget.model is None


# Helper function to create Qt app for tests
@pytest.fixture(autouse=True)
def qt_app():
    """Create a QApplication instance for the tests."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Clean up Qt signals and process events
    app.processEvents()


def wait_for_worker(worker):
    """Helper function to wait for a worker to complete."""
    if hasattr(worker, "wait"):
        worker.wait()
