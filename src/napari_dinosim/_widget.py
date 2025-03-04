import json
import os
from typing import Optional

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    PushButton,
    create_widget,
)
from napari.layers import Image, Points
from napari.qt import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QVBoxLayout,
)
from torch import cuda, device, float32, hub, tensor
from torch.backends import mps

from .dinoSim_pipeline import DinoSim_pipeline
from .utils import gaussian_kernel, get_img_processing_f, torch_convolve

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DINOSim_widget(Container):
    """DINOSim napari widget for zero-shot image segmentation using DINO vision transformers.

    This widget provides a graphical interface for loading DINO models, selecting reference
    points in images, and generating segmentation masks based on visual similarity.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance this widget will be attached to.

    Attributes
    ----------
    compute_device : torch.device
        The device (CPU/GPU/MPS) used for computation.
    model_dims : dict
        Dictionary mapping model sizes to their number of feature dimensions.
    crop_sizes : dict
        Dictionary mapping scale factors to crop dimensions.
    model : torch.nn.Module
        The loaded DINO vision transformer model.
    feat_dim : int
        Feature dimension of the current model.
    pipeline_engine : DinoSim_pipeline
        The processing pipeline for computing embeddings and similarities.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        if cuda.is_available():
            compute_device = device("cuda")
        elif mps.is_available():
            compute_device = device("mps")
        else:
            compute_device = device("cpu")
        self._viewer = viewer
        self.compute_device = compute_device
        self.model_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.crop_sizes = {
            "x1": (518, 518),
            "x0.5": (1036, 1036),
            "x2": (259, 259),
        }
        self.model = None
        self.feat_dim = 0
        self.pipeline_engine = None
        self.upsample = "bilinear"  # bilinear, None
        self.resize_size = 518  # should be multiple of model patch_size
        kernel = gaussian_kernel(size=3, sigma=1)
        kernel = tensor(kernel, dtype=float32, device=self.compute_device)
        self.filter = lambda x: torch_convolve(x, kernel)  # gaussian filter
        self._points_layer: Optional[napari.layers.Points] = None
        self.loaded_img_layer: Optional[napari.layers.Image] = None
        # Store active workers to prevent premature garbage collection
        self._active_workers = []

        # Show welcome dialog with instructions
        self._show_welcome_dialog()

        # GUI elements
        self._create_gui()

        # Variables to store intermediate results
        self._references_coord = []
        self.predictions = None
        self._load_model()

    def _show_welcome_dialog(self):
        """Show welcome dialog with usage instructions."""

        # Check if user has chosen to hide dialog
        hide_file = os.path.join(
            os.path.expanduser("~"), ".dinosim_preferences"
        )
        if os.path.exists(hide_file):
            with open(hide_file) as f:
                preferences = json.load(f)
                if preferences.get("hide_welcome", False):
                    return

        dialog = QDialog()
        dialog.setWindowTitle("Welcome to DINOSim")
        layout = QVBoxLayout()

        # Add usage instructions
        instructions = """
        <h3>Welcome to DINOSim!</h3>
        <p>Quick start guide:</p>
        <ol>
            <li>Drag and drop your image into the viewer</li>
            <li>Click on the regions of interest in your image to set reference points</li>
        </ol>
        <p>
        The smallest model is loaded by default for faster processing.
        To use a different model size, select it from the dropdown and click 'Load Model'.
        Larger models may provide better results but require more computational resources.
        </p>
        <p>
        You can adjust processing parameters in the right menu to optimize results for your data.
        </p>
        """
        label = QLabel(instructions)
        label.setWordWrap(True)
        layout.addWidget(label)

        # Add checkbox for auto-hide option
        hide_checkbox = QCheckBox("Don't show this message again")
        layout.addWidget(hide_checkbox)

        # Add OK button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        def save_preference():
            if hide_checkbox.isChecked():
                # Create hide file to store preference
                os.makedirs(os.path.dirname(hide_file), exist_ok=True)
                with open(hide_file, "w") as f:
                    json.dump({"hide_welcome": True}, f, indent=4)

        # Connect to accepted signal
        dialog.accepted.connect(save_preference)

        dialog.setLayout(layout)
        dialog.exec_()

    def _create_gui(self):
        """Create and organize the GUI components.

        Creates three main sections:
        1. Title
        2. Model selection controls
        3. Image processing controls

        Each section is separated by a visual divider.
        """
        # Create title label
        title_label = Label(value="DINOSim")
        title_label.native.setStyleSheet(
            "font-weight: bold; font-size: 18px; qproperty-alignment: AlignCenter;"
        )

        model_section = self._create_model_section()
        processing_section = self._create_processing_section()

        # Create divider labels instead of QFrames
        divider1 = Label(value="─" * 25)  # Using text characters as divider
        divider1.native.setStyleSheet("color: gray;")

        divider2 = Label(value="─" * 25)  # Using text characters as divider
        divider2.native.setStyleSheet("color: gray;")

        # Organize the main container
        self.extend(
            [
                title_label,
                model_section,
                divider1,
                processing_section,
            ]
        )

    def _create_model_section(self):
        """Create the model selection section of the GUI.

        Returns
        -------
        Container
            A widget container with model size selector, load button, and GPU status.
        """
        model_section_label = Label(
            value="Model Selection", name="section_label"
        )
        model_section_label.native.setStyleSheet(
            "font-weight: bold; font-size: 14px;"
        )

        model_size_label = Label(value="Model Size:", name="subsection_label")
        self.model_size_selector = ComboBox(
            value="small",
            choices=list(self.model_dims.keys()),
            tooltip="Select the model size (s=small, b=base, l=large, g=giant). Larger models may be more accurate but require more resources.",
        )
        model_size_container = Container(
            widgets=[model_size_label, self.model_size_selector],
            layout="horizontal",
            labels=False,
        )

        self._load_model_btn = PushButton(
            text="Load Model",
            tooltip="Download (if necessary) and load selected model.",
        )
        self._load_model_btn.changed.connect(self._load_model)
        self._load_model_btn.native.setStyleSheet(
            "background-color: red; color: black;"
        )

        gpu_label = Label(value="GPU Status:", name="subsection_label")
        self._notification_checkbox = CheckBox(
            text=(
                "Available"
                if cuda.is_available() or mps.is_available()
                else "Not Available"
            ),
            value=cuda.is_available(),
        )
        self._notification_checkbox.enabled = False
        self._notification_checkbox.native.setStyleSheet(
            "QCheckBox::indicator { width: 20px; height: 20px; }"
        )
        gpu_container = Container(
            widgets=[gpu_label, self._notification_checkbox],
            layout="horizontal",
            labels=False,
        )

        return Container(
            widgets=[
                model_section_label,
                model_size_container,
                self._load_model_btn,
                gpu_container,
            ],
            labels=False,
        )

    def _create_processing_section(self):
        """Create the image processing section of the GUI.

        Returns
        -------
        Container
            A widget container with reference controls, image selection,
            crop size selector, and threshold controls.
        """
        image_section_label = Label(value="Settings", name="section_label")
        image_section_label.native.setStyleSheet(
            "font-weight: bold; font-size: 14px;"
        )

        # Reference controls container
        ref_controls = self._create_reference_controls()

        image_layer_label = Label(
            value="Image to segment:", name="subsection_label"
        )
        self._image_layer_combo = create_widget(
            annotation="napari.layers.Image"
        )
        self._image_layer_combo.native.setStyleSheet(
            "QComboBox { max-width: 200px; }"
        )
        self._image_layer_combo.reset_choices()
        self._image_layer_combo.changed.connect(self._new_image_selected)

        # Connect to layer name changes
        def _on_layer_name_changed(event):
            self._image_layer_combo.reset_choices()
            # Maintain the current selection if possible
            if event.source in self._viewer.layers:
                self._image_layer_combo.value = event.source

        # Connect to name changes for all existing layers
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.events.name.connect(_on_layer_name_changed)

        # Update connection in layer insertion handler
        def _connect_layer_name_change(layer):
            if isinstance(layer, Image):
                layer.events.name.connect(_on_layer_name_changed)

        self._viewer.layers.events.inserted.connect(
            lambda e: _connect_layer_name_change(e.value)
        )

        image_layer_container = Container(
            widgets=[image_layer_label, self._image_layer_combo],
            layout="horizontal",
            labels=False,
        )
        self._points_layer = None

        crop_size_label = Label(
            value="Scaling Factor:", name="subsection_label"
        )
        self.crop_size_selector = ComboBox(
            value="x1",
            choices=list(self.crop_sizes.keys()),
            tooltip="Select scaling factor. The smaller the crop the larger the zoom.",
        )
        self.crop_size_selector.changed.connect(self._new_crop_size_selected)
        crop_size_container = Container(
            widgets=[crop_size_label, self.crop_size_selector],
            layout="horizontal",
            labels=False,
        )

        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        threshold_label = Label(
            value="Segmentation Threshold:", name="subsection_label"
        )
        self._threshold_slider = create_widget(
            annotation=float,
            widget_type="FloatSlider",
            value=0.5,
        )
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        self._threshold_slider.changed.connect(self._threshold_im)
        threshold_container = Container(
            widgets=[threshold_label, self._threshold_slider], labels=False
        )

        self._reset_btn = PushButton(
            text="Reset Default Settings",
            tooltip="Reset references and embeddings.",
        )
        self._reset_btn.changed.connect(self.reset_all)

        return Container(
            widgets=[
                image_section_label,
                ref_controls,
                image_layer_container,
                crop_size_container,
                threshold_container,
                self._reset_btn,
            ],
            labels=False,
        )

    def _create_reference_controls(self):
        """Create controls for managing reference points and embeddings.

        Returns
        -------
        Container
            A widget container with reference information display and save/load buttons.
        """
        # Reference information labels
        ref_image_label = Label(
            value="Reference Image:", name="subsection_label"
        )
        self._ref_image_name = Label(value="None", name="info_label")
        self._ref_image_name.native.setStyleSheet("max-width: 150px;")
        self._ref_image_name.native.setWordWrap(False)
        ref_image_container = Container(
            widgets=[ref_image_label, self._ref_image_name],
            layout="horizontal",
            labels=False,
        )

        ref_points_label = Label(
            value="Reference Points:", name="subsection_label"
        )
        self._ref_points_name = Label(value="None", name="info_label")
        self._ref_points_name.native.setStyleSheet("max-width: 150px;")
        self._ref_points_name.native.setWordWrap(False)
        ref_points_container = Container(
            widgets=[ref_points_label, self._ref_points_name],
            layout="horizontal",
            labels=False,
        )

        # Save/Load reference buttons
        self._save_ref_btn = PushButton(
            text="Save Reference",
            tooltip="Save current reference to a file",
        )
        self._save_ref_btn.changed.connect(self._save_reference)

        self._load_ref_btn = PushButton(
            text="Load Reference",
            tooltip="Load reference from a file",
        )
        self._load_ref_btn.changed.connect(self._load_reference)

        ref_buttons = Container(
            widgets=[self._save_ref_btn, self._load_ref_btn],
            layout="horizontal",
            labels=False,
        )

        return Container(
            widgets=[ref_image_container, ref_points_container, ref_buttons],
            labels=False,
        )

    def _save_reference(self):
        """Save the current reference to a file."""
        if (
            self.pipeline_engine is None
            or not self.pipeline_engine.exist_reference
        ):
            self._viewer.status = "No reference to save"
            return

        filepath, _ = QFileDialog.getSaveFileName(
            None, "Save Reference", "", "Reference Files (*.pt)"
        )

        if filepath:
            if not filepath.endswith(".pt"):
                filepath += ".pt"
            try:
                self.pipeline_engine.save_reference(filepath)
                self._viewer.status = f"Reference saved to {filepath}"
            except Exception as e:
                self._viewer.status = f"Error saving reference: {str(e)}"

    def _load_reference(self):
        """Load reference from a file."""
        if self.pipeline_engine is None:
            self._viewer.status = "Model not loaded"
            return

        filepath, _ = QFileDialog.getOpenFileName(
            None, "Load Reference", "", "Reference Files (*.pt)"
        )

        if filepath:
            try:
                self.pipeline_engine.load_reference(
                    filepath, filter=self.filter
                )
                self._ref_image_name.value = "Loaded reference"
                self._ref_points_name.value = "Loaded reference"
                self._get_dist_map()
                self._viewer.status = f"Reference loaded from {filepath}"
            except Exception as e:
                self._viewer.status = f"Error loading reference: {str(e)}"

    def _new_image_selected(self):
        if self.pipeline_engine is None:
            return
        self.pipeline_engine.delete_precomputed_embeddings()
        self.auto_precompute()
        self._get_dist_map()

    def _start_worker(
        self, worker, finished_callback=None, cleanup_callback=None
    ):
        """Start a worker thread with proper cleanup.

        Parameters
        ----------
        worker : FunctionWorker
            The worker to start
        finished_callback : callable, optional
            Callback to run when worker finishes successfully
        cleanup_callback : callable, optional
            Callback to run during cleanup (after finished/errored)
        """

        def _cleanup():
            try:
                if worker in self._active_workers:
                    self._active_workers.remove(worker)
                if cleanup_callback:
                    cleanup_callback()
            except RuntimeError:
                # Handle case where Qt C++ object was deleted
                pass

        def _on_finished():
            try:
                if finished_callback:
                    finished_callback()
            finally:
                _cleanup()

        def _on_errored(e):
            try:
                print(f"Worker error: {str(e)}")  # Log the error for debugging
            finally:
                _cleanup()

        # Keep strong references to callbacks to prevent premature garbage collection
        worker._cleanup_func = _cleanup
        worker._finished_func = _on_finished
        worker._errored_func = _on_errored

        worker.finished.connect(_on_finished)
        worker.errored.connect(_on_errored)
        self._active_workers.append(worker)
        worker.start()

    def _new_crop_size_selected(self):
        self._reset_emb_and_ref()
        worker = self.precompute_threaded()
        self._start_worker(
            worker, finished_callback=self._update_reference_and_process
        )

    def _check_existing_image_and_preprocess(self):
        """Check for existing image layers and preprocess if found."""
        image_found = False
        points_found = False
        for layer in self._viewer.layers:
            if not image_found and isinstance(layer, Image):
                self._image_layer_combo.value = layer
                self.auto_precompute()
                image_found = True
                # Process the first found image layer

            if not points_found and isinstance(layer, Points):
                self._points_layer = layer
                self._points_layer.events.data.connect(
                    self._update_reference_and_process
                )
                points_found = True
                # Process the first found points layer

            if image_found and points_found:
                self._update_reference_and_process()
                break

        if image_found and not points_found:
            self._add_points_layer()

    @thread_worker()
    def precompute_threaded(self):
        self.auto_precompute()

    def auto_precompute(self):
        """Automatically precompute embeddings for the current image.

        Processes the currently selected image layer to compute DINO embeddings
        if they haven't been computed already. Handles both grayscale and RGB images.
        """
        if self.pipeline_engine is not None:
            image_layer = self._image_layer_combo.value  # (n),h,w,(c)
            if image_layer is not None:
                image = self._get_nhwc_image(image_layer.data)
                assert image.shape[-1] in [
                    1,
                    3,
                    4,
                ], f"{image.shape[-1]} channels are not allowed, only 1, 3 or 4"
                # image = ((image / np.iinfo(image.dtype).max) * 255).astype(np.uint8)
                image = self._touint8(image)
                if not self.pipeline_engine.emb_precomputed:
                    self.loaded_img_layer = self._image_layer_combo.value
                    crop_x, crop_y = self.crop_sizes[
                        self.crop_size_selector.value
                    ]
                    self.pipeline_engine.pre_compute_embeddings(
                        image,
                        overlap=(0, 0),
                        padding=(0, 0),
                        crop_shape=(crop_x, crop_y, image.shape[-1]),
                        verbose=True,
                        batch_size=1,
                    )

    def _touint8(self, image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 format with proper normalization.

        Parameters
        ----------
        image : np.ndarray
            Input image array. Can be float (0-1 or arbitrary range) or int.

        Returns
        -------
        np.ndarray
            Converted uint8 image with values 0-255.
        """
        if image.dtype != np.uint8:
            if image.min() >= 0 and image.max() <= 255:
                pass
            else:
                if not (0 <= image.min() <= 1 and 0 <= image.max() <= 1):
                    image = image - image.min()
                    image = image / image.max()
                image = image * 255
        return image.astype(np.uint8)

    def _get_nhwc_image(self, image):
        """Convert image to NHWC format (batch, height, width, channels).

        Parameters
        ----------
        image : np.ndarray
            Input image array.

        Returns
        -------
        np.ndarray
            Image in NHWC format with explicit batch and channel dimensions.
        """
        image = np.squeeze(image)
        if len(image.shape) == 2:
            image = image[np.newaxis, ..., np.newaxis]
        elif len(image.shape) == 3:
            if image.shape[-1] in [3, 4]:
                # consider (h,w,c) rgb or rgba
                image = image[np.newaxis, ..., :3]  # remove possible alpha
            else:
                # consider 3D (n,h,w)
                image = image[..., np.newaxis]
        return image

    def _reset_emb_and_ref(self):
        if self.pipeline_engine is not None:
            self.pipeline_engine.delete_references()
            self.pipeline_engine.delete_precomputed_embeddings()
            # Reset reference information labels
            self._ref_image_name.value = "None"
            self._ref_points_name.value = "None"

    def reset_all(self):
        """Reset references and embeddings."""
        if self.pipeline_engine is not None:
            self._threshold_slider.value = 0.5
            self._reset_emb_and_ref()
            worker = self.precompute_threaded()
            worker.start()

    def _get_dist_map(self, apply_threshold=True):
        """Generate and display the thresholded distance map."""
        if self.pipeline_engine is None:
            self._viewer.status = "Model not loaded"
            return

        if not self.pipeline_engine.exist_reference:
            self._viewer.status = "No reference points selected"
            return

        try:
            distances = self.pipeline_engine.get_ds_distances_sameRef(
                verbose=False
            )
            self.predictions = self.pipeline_engine.distance_post_processing(
                distances,
                self.filter,
                upsampling_mode=self.upsample,
            )
            if apply_threshold:
                self._threshold_im()
        except Exception as e:
            self._viewer.status = f"Error processing image: {str(e)}"

    def _threshold_im(self):
        # simple callback, otherwise numeric value is given as parameter
        self.threshold_im()

    def threshold_im(self, file_name=None):
        """Apply threshold to prediction map and display result.

        Parameters
        ----------
        file_name : str, optional
            Base name for the output mask layer. If None, uses input image name.
        """
        if self.predictions is not None:
            thresholded = self.predictions < self._threshold_slider.value
            thresholded = np.squeeze(thresholded * 255).astype(np.uint8)
            name = (
                self._image_layer_combo.value.name
                if file_name is None
                else file_name
            )
            name += "_mask"

            if name in self._viewer.layers:
                self._viewer.layers[name].data = thresholded
            else:
                self._viewer.add_labels(thresholded, name=name)

    def _update_reference_and_process(self):
        """Update reference coordinates and process the image.

        Gets points from the points layer, updates reference vectors,
        and triggers recomputation of the similarity map.
        """
        points_layer = self._points_layer
        if points_layer is None:
            return

        image_layer = self._image_layer_combo.value
        if image_layer is not None:
            # Update reference information labels
            self._ref_image_name.value = image_layer.name
            self._ref_points_name.value = (
                points_layer.name if points_layer else "None"
            )

            image = self._get_nhwc_image(image_layer.data)
            points = np.array(points_layer.data, dtype=int)
            n, h, w, c = image.shape
            # Compute mean color of the selected points
            self._references_coord = []
            for point in points:
                z, y, x = point if n > 1 else (0, *point)  # Handle 3D and 2D
                if 0 <= x < w and 0 <= y < h and 0 <= z < n:
                    self._references_coord.append((z, x, y))

            if (
                self.pipeline_engine is not None
                and len(self._references_coord) > 0
            ):
                worker = self.precompute_threaded()
                self._start_worker(worker)
                self.pipeline_engine.set_reference_vector(
                    list_coords=self._references_coord, filter=self.filter
                )
                self._get_dist_map()

    def _load_model(self):
        self._image_layer_combo.reset_choices()
        worker = self._load_model_threaded()
        self._start_worker(
            worker, finished_callback=self._check_existing_image_and_preprocess
        )

    @thread_worker()
    def _load_model_threaded(self):
        """Load the selected model based on the user's choice."""
        try:
            model_size = self.model_size_selector.value
            model_letter = model_size[0]

            if self.feat_dim != self.model_dims[model_size]:
                if self.model is not None:
                    self.model = None
                    cuda.empty_cache()

                self._load_model_btn.native.setStyleSheet(
                    "background-color: yellow; color: black;"
                )
                self._load_model_btn.text = "Loading model..."

                self.model = hub.load(
                    "facebookresearch/dinov2",
                    f"dinov2_vit{model_letter}14_reg",
                )
                self.model.to(self.compute_device)
                self.model.eval()

                self.feat_dim = self.model_dims[model_size]

                self._load_model_btn.native.setStyleSheet(
                    "background-color: lightgreen; color: black;"
                )
                self._load_model_btn.text = (
                    f"Load New Model\n(Current Model: {model_size})"
                )

                if self.pipeline_engine is not None:
                    self.pipeline_engine = None

                self.pipeline_engine = DinoSim_pipeline(
                    self.model,
                    self.model.patch_size,
                    self.compute_device,
                    get_img_processing_f(resize_size=self.resize_size),
                    self.feat_dim,
                    dino_image_size=self.resize_size,
                )
        except Exception as e:
            self._viewer.status = f"Error loading model: {str(e)}"

    def _add_points_layer(self):
        """Add points layer only if no reference is loaded."""
        # Skip if reference is already loaded
        if (
            self.pipeline_engine is not None
            and self.pipeline_engine.exist_reference
        ):
            return

        if self._points_layer is None:
            # Check if the loaded image layer is 3D
            image_layer = self._image_layer_combo.value
            # Check actual dimensionality of the layer
            if image_layer is not None and image_layer.ndim > 2:
                # Create a 3D points layer
                points_layer = self._viewer.add_points(
                    data=None, size=10, name="Points Layer", ndim=3
                )
            else:
                # Create a 2D points layer
                points_layer = self._viewer.add_points(
                    data=None, size=10, name="Points Layer"
                )

            points_layer.mode = "add"
            self._viewer.layers.selection.active = self._viewer.layers[
                "Points Layer"
            ]

    def _on_layer_inserted(self, event):
        try:
            layer = event.value

            if isinstance(layer, Image):
                # Reset choices before setting new value
                self._image_layer_combo.reset_choices()
                self._image_layer_combo.value = layer

                # Start precomputation
                worker = self.precompute_threaded()

                if self.pipeline_engine:
                    if self.pipeline_engine.exist_reference:
                        # If reference exists, automatically process the image
                        self._start_worker(
                            worker, finished_callback=self._get_dist_map
                        )
                    else:
                        # If no reference, add points layer as before
                        self._start_worker(
                            worker, finished_callback=self._add_points_layer
                        )
                else:
                    self._start_worker(worker)

            elif isinstance(layer, Points):
                if self._points_layer is not None:
                    self._points_layer.events.data.disconnect(
                        self._update_reference_and_process
                    )
                layer.mode = "add"
                self._points_layer = layer
                self._points_layer.events.data.connect(
                    self._update_reference_and_process
                )
        except Exception as e:
            print(e)
            self._viewer.status = f"Error: {str(e)}"

    def _on_layer_removed(self, event):
        layer = event.value

        if isinstance(layer, Image):
            # Disconnect name change handler
            try:
                layer.events.name.disconnect()
            except TypeError:
                pass  # Handler was already disconnected

            if self.pipeline_engine != None and self.loaded_img_layer == layer:
                self.pipeline_engine.delete_precomputed_embeddings()
                self.loaded_img_layer = ""
            self._image_layer_combo.reset_choices()

        elif layer is self._points_layer:
            self._points_layer.events.data.disconnect(
                self._update_reference_and_process
            )
            self._points_layer = None
            if self.pipeline_engine != None:
                self.pipeline_engine.delete_references()

    def closeEvent(self, event):
        """Clean up resources when widget is closed."""
        try:
            # Make a copy of the list since we'll be modifying it during iteration
            workers = self._active_workers[:]
            for worker in workers:
                try:
                    if hasattr(worker, "quit"):
                        worker.quit()
                    if hasattr(worker, "wait"):
                        worker.wait()  # Wait for worker to finish
                    # Disconnect all signals
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
                    # Handle case where Qt C++ object was deleted
                    pass
                if worker in self._active_workers:
                    self._active_workers.remove(worker)

            if self.pipeline_engine is not None:
                del self.pipeline_engine
                self.pipeline_engine = None

            if self.model is not None:
                del self.model
                self.model = None

            # Clear any remaining references
            self._active_workers.clear()

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            super().closeEvent(event)
