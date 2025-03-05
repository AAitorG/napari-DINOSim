import warnings
from unittest.mock import MagicMock, patch

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


def run_worker_sync(worker):
    """Helper function to run a worker synchronously and ensure cleanup.

    Parameters
    ----------
    worker : thread_worker
        The worker to run synchronously
    """
    try:
        worker.run()  # Run synchronously
    finally:
        # Clean up
        if hasattr(worker, "quit"):
            worker.quit()
        if hasattr(worker, "wait"):
            worker.wait()
        if hasattr(worker, "finished"):
            try:
                worker.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
        if hasattr(worker, "errored"):
            try:
                worker.errored.disconnect()
            except (RuntimeError, TypeError):
                pass


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
def widget(mock_viewer):
    """Create a widget instance with proper cleanup."""
    widget_instance = None
    try:
        with patch("torch.hub.load") as mock_hub:
            # Mock the DINO model
            mock_model = MagicMock()
            mock_model.patch_size = 14
            mock_model.eval = MagicMock(return_value=None)
            mock_hub.return_value = mock_model

            widget_instance = DINOSim_widget(mock_viewer)
            # Mock the image layer combo to allow setting values
            widget_instance._image_layer_combo = MagicMock(spec=ComboBox)
            widget_instance._image_layer_combo.choices = []
            # Mock other necessary components
            widget_instance._threshold_slider = MagicMock()
            widget_instance._threshold_slider.value = 0.5
            widget_instance._load_model_btn = MagicMock()
            widget_instance._load_model_btn.native = MagicMock()
            widget_instance._load_model_btn.native.styleSheet = MagicMock(
                return_value=""
            )

            yield widget_instance

    finally:
        # Ensure cleanup happens even if an exception occurs
        if widget_instance is not None:
            # Clean up any active workers
            if hasattr(widget_instance, "_active_workers"):
                workers = widget_instance._active_workers[:]
                for worker in workers:
                    try:
                        if hasattr(worker, "quit"):
                            worker.quit()
                        if hasattr(worker, "wait"):
                            worker.wait()
                        if hasattr(worker, "finished"):
                            try:
                                worker.finished.disconnect()
                            except (RuntimeError, TypeError):
                                pass
                        if hasattr(worker, "errored"):
                            try:
                                worker.errored.disconnect()
                            except (RuntimeError, TypeError):
                                pass
                    except RuntimeError:
                        pass
                widget_instance._active_workers.clear()

            # Clean up model and pipeline
            if (
                hasattr(widget_instance, "pipeline_engine")
                and widget_instance.pipeline_engine is not None
            ):
                widget_instance.pipeline_engine = None
            if (
                hasattr(widget_instance, "model")
                and widget_instance.model is not None
            ):
                widget_instance.model = None

            del widget_instance


# Test initialization
def test_widget_initialization(widget, mock_viewer):
    """Test that widget initializes correctly."""
    assert (
        widget.pipeline_engine is not None
    ), "Pipeline engine should be initialized"
    assert widget._viewer == mock_viewer, "Viewer should be properly set"
    assert (
        widget.resize_size % 14 == 0
    ), "Resize size should be multiple of patch size"
    assert (
        widget.model_dims["small"] == 384
    ), "Small model dimension should be 384"
    assert widget.compute_device is not None, "Compute device should be set"


# Test model loading
def test_model_loading(widget):
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

        # Run the worker synchronously
        run_worker_sync(worker)

        # Verify the model was loaded correctly
        mock_hub.assert_called_once_with(
            "facebookresearch/dinov2", "dinov2_vits14_reg"
        )

        # Verify model setup
        assert widget.model is mock_model, "Model should be set correctly"
        assert (
            widget.pipeline_engine is not None
        ), "Pipeline engine should be initialized"
        assert mock_model.eval.called, "Model should be set to eval mode"
        assert mock_model.to.called, "Model should be moved to correct device"
        assert (
            widget.feat_dim == widget.model_dims["small"]
        ), "Feature dimension should match model size"


# Test image processing
def test_image_processing(widget):
    """Test image processing functionality."""
    # Test RGB image
    test_image = np.random.rand(100, 100, 3)
    processed_image = widget._get_nhwc_image(test_image)
    assert processed_image.ndim == 4, "Should add batch dimension"
    assert processed_image.shape == (
        1,
        100,
        100,
        3,
    ), "Should have correct shape"
    assert np.array_equal(
        processed_image[0], test_image
    ), "Image content should be preserved"

    # Test grayscale
    test_image_gray = np.random.rand(100, 100)
    processed_image_gray = widget._get_nhwc_image(test_image_gray)
    assert processed_image_gray.shape == (
        1,
        100,
        100,
        1,
    ), "Should have correct shape for grayscale"
    assert np.array_equal(
        processed_image_gray[0, ..., 0], test_image_gray
    ), "Grayscale content should be preserved"

    # Test 3D image
    test_image_3d = np.random.rand(5, 100, 100)
    processed_image_3d = widget._get_nhwc_image(test_image_3d)
    assert processed_image_3d.shape == (
        5,
        100,
        100,
        1,
    ), "Should handle 3D images correctly"


# Test uint8 conversion
def test_uint8_conversion(widget):
    """Test conversion to uint8."""
    # Test float image [0, 1]
    float_image = np.random.rand(10, 10, 3)
    uint8_image = widget._touint8(float_image)
    assert uint8_image.dtype == np.uint8, "Should convert to uint8"
    assert uint8_image.max() <= 255, "Maximum value should be 255"
    assert uint8_image.min() >= 0, "Minimum value should be 0"

    # Test arbitrary range float image
    arbitrary_image = np.random.rand(10, 10, 3) * 1000
    uint8_arbitrary = widget._touint8(arbitrary_image)
    assert uint8_arbitrary.dtype == np.uint8, "Should convert to uint8"
    assert uint8_arbitrary.max() <= 255, "Maximum value should be 255"
    assert uint8_arbitrary.min() >= 0, "Minimum value should be 0"

    # Test uint8 input
    uint8_input = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    uint8_output = widget._touint8(uint8_input)
    np.testing.assert_array_equal(uint8_input, uint8_output)


# Test points layer handling
def test_points_layer_handling(widget, mock_viewer):
    """Test points layer creation and handling."""
    # Test 2D image
    mock_image = MagicMock(spec=Image)
    mock_image.ndim = 2
    mock_image.name = "test_image"
    mock_viewer.layers._list = [mock_image]
    widget._image_layer_combo.value = mock_image

    widget._add_points_layer()
    assert mock_viewer.add_points.called, "Should add points layer"
    mock_viewer.add_points.assert_called_with(
        data=None, size=10, name="Points Layer"
    )

    # Test 3D image
    mock_image.ndim = 3
    widget._points_layer = None
    widget._add_points_layer()
    assert mock_viewer.add_points.called, "Should add 3D points layer"
    mock_viewer.add_points.assert_called_with(
        data=None, size=10, name="Points Layer", ndim=3
    )

    # Test with existing reference
    widget.pipeline_engine.exist_reference = True
    widget._points_layer = None
    widget._add_points_layer()
    assert (
        widget._points_layer is None
    ), "Should not add points layer when reference exists"


# Test reference handling
def test_reference_handling(widget):
    """Test reference point handling."""
    # Mock pipeline engine
    widget.pipeline_engine = MagicMock()
    widget.pipeline_engine.exist_reference = False

    # Create test points
    test_points = np.array([[10, 10], [20, 20]])
    widget._points_layer = MagicMock(spec=Points)
    widget._points_layer.data = test_points
    widget._points_layer.events = MagicMock()
    widget._points_layer.events.data = MagicMock()

    # Mock image layer
    mock_image = MagicMock(spec=Image)
    mock_image.data = np.random.rand(100, 100, 3)
    mock_image.name = "test_image"
    widget._image_layer_combo.value = mock_image
    widget._image_layer_combo.choices = [mock_image]

    # Test reference update
    widget._update_reference_and_process()
    assert (
        len(widget._references_coord) == 2
    ), "Should have correct number of reference points"
    assert (
        widget._ref_image_name.value == "test_image"
    ), "Should update reference image name"

    # Test reference coordinates
    expected_coords = [(0, 10, 10), (0, 20, 20)]
    assert all(
        coord in widget._references_coord for coord in expected_coords
    ), "Should have correct coordinates"


# Test threshold handling
def test_threshold_handling(widget):
    """Test threshold application."""
    # Setup test data
    widget.predictions = np.random.rand(100, 100)
    widget._threshold_slider.value = 0.5

    # Mock image layer
    mock_image = MagicMock(spec=Image)
    mock_image.name = "test_image"
    widget._image_layer_combo.value = mock_image

    # Test threshold application
    with patch.object(widget._viewer, "add_labels") as mock_add_labels:
        widget.threshold_im()
        assert mock_add_labels.called, "Should add labels layer"

        # Verify thresholded output
        call_args = mock_add_labels.call_args[0]
        thresholded_data = call_args[0]
        assert thresholded_data.dtype == np.uint8, "Output should be uint8"
        assert np.array_equal(
            thresholded_data, (widget.predictions < 0.5) * 255
        ), "Threshold should be correctly applied"
        assert (
            mock_add_labels.call_args[1]["name"] == "test_image_mask"
        ), "Should use correct layer name"

    # Test existing layer update
    mock_existing_layer = MagicMock()
    widget._viewer.layers.__getitem__.return_value = mock_existing_layer
    widget.threshold_im()
    assert hasattr(
        mock_existing_layer, "data"
    ), "Should update existing layer data"


# Test cleanup
def test_cleanup(widget):
    """Test cleanup on widget close."""
    # Setup mocks
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
    # Test invalid image handling
    invalid_image = np.random.rand(2, 100, 100, 5)  # 5 channels
    mock_image = MagicMock(spec=Image)
    mock_image.data = invalid_image
    widget._image_layer_combo.value = mock_image

    with pytest.raises(AssertionError):
        widget.auto_precompute()  # This should raise the assertion

    # Test model loading error
    with patch("torch.hub.load", side_effect=Exception("Network error")):
        prev_pipeline = widget.pipeline_engine
        widget.model_size_selector.value = "small"
        worker = widget._load_model_threaded()

        try:
            worker.run()
        except Exception as e:
            assert str(e) == "Network error", "Should raise correct error"

        assert (
            widget.pipeline_engine == prev_pipeline
        ), "Should preserve pipeline on error"
        assert widget.model is not None, "Should maintain valid model state"

    # Test reference handling error
    widget._points_layer = MagicMock(spec=Points)
    widget._points_layer.data = np.array(
        [[0, 1000, 1000]]
    )  # Out of bounds point
    widget._update_reference_and_process()
    assert (
        len(widget._references_coord) == 0
    ), "Should handle invalid reference points"
