import json
import logging
import os

logger = logging.getLogger(__name__)
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    PushButton,
    create_widget,
    FloatSpinBox,
)
from napari.layers import Image, Points
from napari.qt import thread_worker
from napari.viewer import Viewer

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from .utils import (
    DINOSim_pipeline,
    CollapsibleSection,
    gaussian_kernel,
    get_img_processing_f,
    torch_convolve,
    get_nhwc_image,
)
from ._sam2_widget import SAM2WidgetHelper, HAS_SAM2
from ._embedding_manager import EmbeddingManager
from ._layer_handlers import LayerEventHandler


class DINOSim_widget(QWidget):
    """DINOSim napari widget for zero-shot image segmentation using DINO vision transformers.

    This widget provides a graphical interface for loading DINO models, selecting reference
    points in images, and generating segmentation masks based on visual similarity.
    """

    def __init__(self, viewer: Viewer):
        super().__init__()

        # Create main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create a Container for all content
        self.container = Container(layout="vertical")

        # Create a scroll area and set its properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.container.native)
        scroll_area.setMinimumHeight(400)

        # Add the scroll area to the main layout
        main_layout.addWidget(scroll_area)

        if torch.cuda.is_available():
            compute_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            compute_device = torch.device("mps")
        else:
            compute_device = torch.device("cpu")

        self._viewer = viewer
        self.compute_device = compute_device
        self.sam2_compute_device = compute_device
        self.model_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.base_crop_size = 518
        self.model = None
        self.feat_dim = 0
        self.pipeline_engine = None
        self.upsample = "bilinear"
        self.resize_size = 518

        kernel = gaussian_kernel(size=3, sigma=1)
        kernel = torch.tensor(
            kernel, dtype=torch.float32, device=self.compute_device
        )
        self.filter = lambda x: torch_convolve(x, kernel)

        self._points_layer: Optional[Points] = None
        self.loaded_img_layer: Optional[Image] = None
        self._active_workers = []
        self._is_inserting_layer = False
        self._is_programmatic_scale_change = False
        self._is_programmatic_threshold_change = False

        # Instantiate helpers
        self.has_sam2 = HAS_SAM2
        self.sam2_helper = SAM2WidgetHelper(self)
        self.embedding_manager = EmbeddingManager(self)
        self.layer_handler = LayerEventHandler(self)

        # Show welcome dialog
        self._show_welcome_dialog()

        # Create GUI
        self._create_gui()

        # Results storage
        self._references_coord = []
        self.predictions = None
        self.distances = None

        # Load default model
        self._load_model()

    def _show_welcome_dialog(self):
        """Show welcome dialog with usage instructions."""
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

        hide_checkbox = QCheckBox("Don't show this message again")
        layout.addWidget(hide_checkbox)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)

        def save_preference():
            if hide_checkbox.isChecked():
                os.makedirs(os.path.dirname(hide_file), exist_ok=True)
                with open(hide_file, "w") as f:
                    json.dump({"hide_welcome": True}, f, indent=4)

        dialog.accepted.connect(save_preference)
        dialog.setLayout(layout)
        dialog.exec_()

    def _create_gui(self):
        """Create and organize the GUI components."""
        container_layout = self.container.native.layout()
        if container_layout is None:
            container_layout = QVBoxLayout()
            self.container.native.setLayout(container_layout)
            container_layout.setContentsMargins(0, 0, 0, 0)

        title_label = Label(value="DINOSim")
        title_label.native.setStyleSheet(
            "font-weight: bold; font-size: 18px; qproperty-alignment: AlignCenter;"
        )
        container_layout.addWidget(title_label.native)

        # Model section
        model_content = self._create_model_section()
        model_section = CollapsibleSection("Model Selection")
        model_section.add_widget(model_content.native)
        container_layout.addWidget(model_section)

        # Processing section
        processing_content = self._create_processing_section()
        processing_section = CollapsibleSection("Settings")
        processing_section.add_widget(processing_content.native)
        container_layout.addWidget(processing_section)

        # SAM2 section
        sam2_content = self.sam2_helper.create_sam2_section()
        sam2_section = CollapsibleSection(
            "SAM2 Post-Processing", collapsed=True
        )
        sam2_section.add_widget(sam2_content.native)
        container_layout.addWidget(sam2_section)

        container_layout.addStretch(1)
        self.container.native.setMaximumWidth(450)

    def _create_model_section(self):
        """Create the model selection section of the GUI."""
        model_size_label = Label(value="Model Size:", name="subsection_label")
        self.model_size_selector = ComboBox(
            value="small",
            choices=list(self.model_dims.keys()),
            tooltip="Select the model size. Larger models may be more accurate but require more resources.",
        )
        model_size_container = Container(
            widgets=[model_size_label, self.model_size_selector],
            layout="horizontal",
            labels=False,
        )
        model_size_container.native.setMaximumWidth(400)

        self._load_model_btn = PushButton(
            text="Load Model",
            tooltip="Download (if necessary) and load selected model.",
        )
        self._load_model_btn.changed.connect(self._load_model)
        self._load_model_btn.native.setStyleSheet(
            "background-color: red; color: black;"
        )

        gpu_label = Label(value="GPU Status:", name="subsection_label")
        gpu_available = (
            torch.cuda.is_available() or torch.backends.mps.is_available()
        )
        self._notification_checkbox = CheckBox(
            text="Available" if gpu_available else "Not Available",
            value=gpu_available,
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
        gpu_container.native.setMaximumWidth(400)

        return Container(
            widgets=[
                model_size_container,
                self._load_model_btn,
                gpu_container,
            ],
            labels=False,
        )

    def _create_processing_section(self):
        """Create the image processing section of the GUI."""
        ref_controls = self._create_reference_controls()

        image_layer_label = Label(
            value="Image to segment:", name="subsection_label"
        )
        self._image_layer_combo = create_widget(
            annotation="napari.layers.Image"
        )
        self._image_layer_combo.reset_choices()
        self._image_layer_combo.changed.connect(self._new_image_selected)

        # Connect to layer events
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.events.name.connect(
                    self.layer_handler.on_layer_name_changed
                )

        self._viewer.layers.events.inserted.connect(
            lambda e: self.layer_handler.connect_layer_name_change(e.value)
        )
        self._viewer.layers.events.removed.connect(
            lambda e: self.layer_handler.disconnect_layer_name_change(e.value)
        )

        image_layer_container = Container(
            widgets=[
                image_layer_label,
                self._image_layer_combo,
                self.embedding_manager._emb_status_indicator,
            ],
            layout="horizontal",
            labels=False,
        )
        image_layer_container.native.setMaximumWidth(400)

        params_section = CollapsibleSection(
            "Processing Parameters", collapsed=False
        )
        params_content = QWidget()
        params_layout = QVBoxLayout(params_content)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(image_layer_container.native)

        crop_size_label = Label(value="Scale Factor:", name="subsection_label")
        self.scale_factor_selector = FloatSpinBox(
            value=1.0,
            min=0.1,
            max=10.0,
            step=0.1,
            tooltip="Select scaling factor. Higher values result in smaller crops (more zoom).",
        )
        self.scale_factor_selector.changed.connect(
            self._new_scale_factor_selected
        )
        crop_size_container = Container(
            widgets=[crop_size_label, self.scale_factor_selector],
            layout="horizontal",
            labels=False,
        )
        crop_size_container.native.setMaximumWidth(400)
        params_layout.addWidget(crop_size_container.native)

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
            widgets=[threshold_label, self._threshold_slider],
            labels=False,
        )
        threshold_container.native.setMaximumWidth(400)
        params_layout.addWidget(threshold_container.native)

        params_section.add_widget(params_content)

        emb_section = CollapsibleSection("Embedding Controls", collapsed=True)
        emb_content = QWidget()
        emb_layout = QVBoxLayout(emb_content)
        emb_layout.setContentsMargins(0, 0, 0, 0)

        precompute_header = Container(
            widgets=[
                Label(value="Auto Precompute:", name="subsection_label"),
                self.embedding_manager.auto_precompute_checkbox,
            ],
            layout="horizontal",
            labels=False,
        )
        precompute_header.native.setMaximumWidth(400)
        emb_layout.addWidget(precompute_header.native)
        emb_layout.addWidget(
            self.embedding_manager.manual_precompute_btn.native
        )

        emb_buttons = Container(
            widgets=[
                self.embedding_manager.save_emb_btn,
                self.embedding_manager.load_emb_btn,
            ],
            layout="horizontal",
            labels=False,
        )
        emb_buttons.native.setMaximumWidth(400)
        emb_layout.addWidget(emb_buttons.native)
        emb_section.add_widget(emb_content)

        self._viewer.layers.events.inserted.connect(
            self.layer_handler.on_layer_inserted
        )
        self._viewer.layers.events.removed.connect(
            self.layer_handler.on_layer_removed
        )

        self._reset_btn = PushButton(
            text="Reset Default Settings",
            tooltip="Reset references and embeddings.",
        )
        self._reset_btn.changed.connect(self.reset_all)

        return Container(
            widgets=[
                ref_controls,
                Container(widgets=[params_section], labels=False),
                Container(widgets=[emb_section], labels=False),
                self._reset_btn,
            ],
            labels=False,
        )

    def _create_reference_controls(self):
        """Create controls for managing reference points and embeddings."""
        ref_container = Container(layout="vertical", labels=False)
        ref_subsection = CollapsibleSection(
            "Reference Information", collapsed=True
        )

        ref_content_widget = QWidget()
        ref_content_layout = QVBoxLayout(ref_content_widget)
        ref_content_layout.setContentsMargins(0, 0, 0, 0)

        self._ref_image_name = Label(value="None", name="info_label")
        self._ref_image_name.native.setStyleSheet("max-width: 150px;")
        ref_image_container = Container(
            widgets=[
                Label(value="Reference Image:", name="subsection_label"),
                self._ref_image_name,
            ],
            layout="horizontal",
            labels=False,
        )
        ref_image_container.native.setMaximumWidth(400)

        self._ref_points_name = Label(value="None", name="info_label")
        self._ref_points_name.native.setStyleSheet("max-width: 150px;")
        ref_points_container = Container(
            widgets=[
                Label(value="Reference Points:", name="subsection_label"),
                self._ref_points_name,
            ],
            layout="horizontal",
            labels=False,
        )
        ref_points_container.native.setMaximumWidth(400)

        ref_content_layout.addWidget(ref_image_container.native)
        ref_content_layout.addWidget(ref_points_container.native)

        self._save_ref_btn = PushButton(
            text="Save Reference", tooltip="Save current reference to a file"
        )
        self._save_ref_btn.changed.connect(self._save_reference)
        self._load_ref_btn = PushButton(
            text="Load Reference", tooltip="Load reference from a file"
        )
        self._load_ref_btn.changed.connect(self._load_reference)

        ref_buttons = Container(
            widgets=[self._save_ref_btn, self._load_ref_btn],
            layout="horizontal",
            labels=False,
        )
        ref_buttons.native.setMaximumWidth(400)
        ref_content_layout.addWidget(ref_buttons.native)

        ref_subsection.add_widget(ref_content_widget)
        ref_container.append(ref_subsection)
        return ref_container

    def _save_reference(self):
        """Open a save dialog and persist the current reference vector to a .pt file."""
        if (
            self.pipeline_engine is None
            or not self.pipeline_engine.exist_reference
        ):
            self._viewer.status = "No reference to save"
            return
        from qtpy.QtWidgets import QFileDialog

        default_filename = "reference"
        if self._image_layer_combo.value is not None:
            default_filename += f"_{self._image_layer_combo.value.name}"
        default_filename += ".pt"
        filepath, _ = QFileDialog.getSaveFileName(
            None, "Save Reference", default_filename, "Reference Files (*.pt)"
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
        """Open a load dialog and restore a reference vector from a .pt file."""
        if self.pipeline_engine is None:
            self._viewer.status = "Model not loaded"
            return
        from qtpy.QtWidgets import QFileDialog

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
        """Handle a user-driven change in the image layer combo box.

        Invalidates precomputed embeddings for the previous image and, if auto-precompute
        is enabled, starts precomputation for the newly selected image.
        """
        if self._is_inserting_layer:
            return
        if self.pipeline_engine is None:
            self.embedding_manager.set_embedding_status("unavailable")
            return
        self.pipeline_engine.delete_precomputed_embeddings()
        self.embedding_manager.set_embedding_status("unavailable")
        self.sam2_helper.refined_mask = None

        if self._image_layer_combo.value is not None:
            is_precomputed = (
                self.loaded_img_layer is not None
                and self._image_layer_combo.value == self.loaded_img_layer
                and self.pipeline_engine is not None
                and self.pipeline_engine.emb_precomputed
            )
            if is_precomputed:
                self.embedding_manager.set_embedding_status("ready")
            else:
                self.loaded_img_layer = None
                self.embedding_manager.set_embedding_status("unavailable")
                if (
                    self.pipeline_engine is not None
                    and self.embedding_manager.auto_precompute_checkbox.value
                ):
                    self.embedding_manager.auto_precompute()
            self.sam2_helper.refined_mask = None

    def _start_worker(
        self, worker, finished_callback=None, cleanup_callback=None
    ):
        """Start a napari thread worker, tracking it and wiring up completion callbacks.

        Args:
            worker: A napari thread_worker instance to start
            finished_callback: Called on successful completion (after cleanup)
            cleanup_callback: Called on both successful completion and errors
        """

        def _cleanup():
            try:
                if worker in self._active_workers:
                    self._active_workers.remove(worker)
                if cleanup_callback:
                    cleanup_callback()
            except RuntimeError:
                pass

        def _on_finished():
            try:
                if finished_callback:
                    finished_callback()
            finally:
                _cleanup()

        def _on_errored(e):
            try:
                logger.error(f"Worker error: {str(e)}", exc_info=True)
            finally:
                _cleanup()

        worker._cleanup_func = _cleanup
        worker._finished_func = _on_finished
        worker._errored_func = _on_errored
        worker.finished.connect(_on_finished)
        worker.errored.connect(_on_errored)
        self._active_workers.append(worker)
        worker.start()

    def _new_scale_factor_selected(self):
        """Handle a user-driven change in the scale factor spinbox.

        Resets embeddings and references, then restarts precomputation if auto mode is on.
        Programmatic changes (e.g. from loading embeddings) are ignored via the guard flag.
        """
        if self._is_programmatic_scale_change:
            return
        self._reset_emb_and_ref()
        if self.embedding_manager.auto_precompute_checkbox.value:
            self.embedding_manager.start_precomputation(
                finished_callback=self._update_reference_and_process
            )

    def _check_existing_image_and_preprocess(self):
        """Scan existing viewer layers after model load and set up image/points if present.

        Called as the finished callback of _load_model_threaded. Finds the first Image and
        Points layers already in the viewer, connects events, and triggers precomputation.
        If no points layer exists, a new one is created.
        """
        image_found = points_found = False
        for layer in self._viewer.layers:
            if not image_found and isinstance(layer, Image):
                self._image_layer_combo.value = layer
                if (
                    self.pipeline_engine
                    and self.pipeline_engine.emb_precomputed
                ):
                    self.embedding_manager.set_embedding_status("ready")
                else:
                    self.embedding_manager.set_embedding_status("unavailable")
                if self.embedding_manager.auto_precompute_checkbox.value:
                    self.embedding_manager.start_precomputation()
                image_found = True
            if not points_found and isinstance(layer, Points):
                self._points_layer = layer
                self._points_layer.events.data.connect(
                    self._update_reference_and_process
                )
                points_found = True
            if image_found and points_found:
                self._update_reference_and_process()
                break
        if image_found and not points_found:
            self._add_points_layer()

    def _reset_emb_and_ref(self):
        """Delete precomputed embeddings and references without resetting other settings."""
        if self.pipeline_engine is not None:
            self.pipeline_engine.delete_references()
            self.pipeline_engine.delete_precomputed_embeddings()
            self.embedding_manager.set_embedding_status("unavailable")
            self._ref_image_name.value = "None"
            self._ref_points_name.value = "None"

    def reset_all(self):
        """Reset all processing state: threshold, scale factor, embeddings, and references.

        Triggered by the 'Reset Default Settings' button. Also restarts auto-precomputation
        if enabled and updates SAM2 status indicators.
        """
        if self.pipeline_engine is not None:
            self._is_programmatic_threshold_change = True
            self._threshold_slider.value = 0.5
            self._is_programmatic_threshold_change = False
            self._is_programmatic_scale_change = True
            self.scale_factor_selector.value = 1.0
            self._is_programmatic_scale_change = False
            self._reset_emb_and_ref()
            if self.embedding_manager.auto_precompute_checkbox.value:
                self.embedding_manager.start_precomputation()
            self.sam2_helper.refined_mask = None
            if self.has_sam2 and self.sam2_helper.sam2_processor is not None:
                if (
                    self.sam2_helper.enable_sam2_checkbox.value
                    and self.sam2_helper.sam2_processor.exist_predictions()
                ):
                    self.sam2_helper.set_sam2_status("ready")
                else:
                    self.sam2_helper.set_sam2_status("unavailable")
            else:
                self.sam2_helper.set_sam2_status("unavailable")

    def _get_dist_map(self, apply_threshold=True):
        """Compute the similarity distance map from current embeddings and reference.

        Runs the full DINOSim inference pipeline (distance computation + post-processing).
        If SAM2 is enabled and masks are loaded, refines the result in a background thread
        before thresholding.

        Args:
            apply_threshold (bool): If True, call threshold_im() after computing distances.
        """
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
                distances, self.filter, upsampling_mode=self.upsample
            )
            self.sam2_helper.refined_mask = None
            sam2_ready = (
                self.has_sam2
                and self.sam2_helper.enable_sam2_checkbox.value
                and self.sam2_helper.sam2_processor is not None
                and self.sam2_helper.sam2_processor.exist_predictions()
            )
            if sam2_ready:
                worker = self.sam2_helper.refine_with_sam2_threaded()
                self._start_worker(
                    worker,
                    finished_callback=lambda: (
                        self.sam2_helper.set_sam2_status("ready"),
                        self.threshold_im(),
                    ),
                )
            else:
                if apply_threshold:
                    self.threshold_im()
        except Exception as e:
            self._viewer.status = f"Error processing image: {str(e)}"

    def _threshold_im(self):
        """Threshold slider callback; ignores programmatic slider changes."""
        if self._is_programmatic_threshold_change:
            return
        self.threshold_im()

    def threshold_im(self, file_name=None):
        """Apply the current threshold to predictions and update the mask layer.

        Uses SAM2-refined predictions when available and enabled. Values below the
        threshold are set to 255 (foreground) and the result is stored in a Labels layer
        named ``<image_name>_mask``.

        Args:
            file_name (str, optional): Override for the output layer name prefix.
                Defaults to the currently selected image layer name.
        """
        if self.predictions is None:
            return
        use_refined = (
            self.has_sam2
            and self.sam2_helper.enable_sam2_checkbox.value
            and self.sam2_helper.refined_mask is not None
        )
        if use_refined:
            if isinstance(self.sam2_helper.refined_mask, torch.Tensor):
                pred = self.sam2_helper.refined_mask.cpu().numpy().copy()
            elif self.sam2_helper.refined_mask is not None:
                pred = np.array(self.sam2_helper.refined_mask)
            else:
                use_refined = False
        if not use_refined:
            if isinstance(self.predictions, torch.Tensor):
                pred = self.predictions.cpu().numpy().copy()
            else:
                pred = np.array(self.predictions)
        if pred.ndim > 2:
            pred = np.squeeze(pred)
        thresholded = (pred < self._threshold_slider.value).astype(
            np.uint8
        ) * 255
        name = f"{self._image_layer_combo.value.name if file_name is None else file_name}_mask"
        if name in self._viewer.layers:
            self._viewer.layers[name].data = thresholded
        else:
            self._viewer.add_labels(thresholded, name=name)

    def _update_reference_and_process(self):
        """Recompute reference vectors and distance map when points change.

        Called whenever the Points layer data changes. Converts napari point coordinates
        to (n, x, y) tuples, sets the reference vector on the pipeline, and triggers a
        new distance map computation. If embeddings are not yet available and auto-precompute
        is enabled, precomputation is started first.
        """
        if self._points_layer is None:
            return
        image_layer = self._image_layer_combo.value
        if image_layer is not None:
            self._ref_image_name.value = image_layer.name
            self._ref_points_name.value = self._points_layer.name
            image = get_nhwc_image(image_layer.data)
            points = np.array(self._points_layer.data, dtype=int)
            n, h, w, c = image.shape
            self._references_coord = []
            for point in points:
                z, y, x = point if n > 1 else (0, *point)
                if 0 <= x < w and 0 <= y < h and 0 <= z < n:
                    self._references_coord.append((z, x, y))
            if (
                self.pipeline_engine is not None
                and len(self._references_coord) > 0
            ):

                def after_precomputation():
                    self.pipeline_engine.set_reference_vector(
                        list_coords=self._references_coord, filter=self.filter
                    )
                    self._get_dist_map()

                if not self.pipeline_engine.emb_precomputed:
                    if self.embedding_manager.auto_precompute_checkbox.value:
                        self.embedding_manager.start_precomputation(
                            finished_callback=after_precomputation
                        )
                    else:
                        self._viewer.status = "Precomputation needed. Use the 'Precompute Now' button."
                else:
                    after_precomputation()

    def _load_model(self):
        """Download (if needed) and load the selected DINOv2 model in a background thread."""
        self._image_layer_combo.reset_choices()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            worker = self._load_model_threaded()
            self._start_worker(
                worker,
                finished_callback=self._check_existing_image_and_preprocess,
            )
        except Exception as e:
            self._viewer.status = f"Error loading model: {str(e)}"

    @thread_worker()
    def _load_model_threaded(self):
        """Worker thread that loads the DINOv2 model and rebuilds the pipeline engine."""
        try:
            model_size = self.model_size_selector.value
            model_letter = model_size[0]
            if self.feat_dim != self.model_dims[model_size]:
                if self.model is not None:
                    self.model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self._load_model_btn.native.setStyleSheet(
                    "background-color: yellow; color: black;"
                )
                self._load_model_btn.text = "Loading model..."
                self.model = torch.hub.load(
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
                    f"Load New Model\n(Current: {model_size})"
                )
                if self.pipeline_engine is not None:
                    self.pipeline_engine = None
                interpolation = (
                    InterpolationMode.BILINEAR
                    if torch.backends.mps.is_available()
                    else InterpolationMode.BICUBIC
                )
                self.pipeline_engine = DINOSim_pipeline(
                    self.model,
                    self.model.patch_size,
                    self.compute_device,
                    get_img_processing_f(
                        resize_size=self.resize_size,
                        interpolation=interpolation,
                    ),
                    self.feat_dim,
                    dino_image_size=self.resize_size,
                )
        except Exception as e:
            self._viewer.status = f"Error loading model: {str(e)}"

    def _add_points_layer(self):
        """Add a new Points layer in 'add' mode if none already exists and no reference is set."""
        if (
            self.pipeline_engine is not None
            and self.pipeline_engine.exist_reference
        ):
            return
        if self._points_layer is None:
            image_layer = self._image_layer_combo.value
            ndim = 3 if image_layer is not None and image_layer.ndim > 2 else 2
            points_layer = self._viewer.add_points(
                data=None, size=10, name="Points Layer", ndim=ndim
            )
            points_layer.mode = "add"
            self._viewer.layers.selection.active = self._viewer.layers[
                "Points Layer"
            ]

    def closeEvent(self, event):
        """Clean up background workers and free GPU memory when the widget is closed."""
        try:
            workers = self._active_workers[:]
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
                if worker in self._active_workers:
                    self._active_workers.remove(worker)
            if self.pipeline_engine is not None:
                del self.pipeline_engine
                self.pipeline_engine = None
            if self.model is not None:
                del self.model
                self.model = None
            if self.sam2_helper.sam2_processor is not None:
                del self.sam2_helper.sam2_processor
                self.sam2_helper.refined_mask = None
            self._active_workers.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        finally:
            QWidget.closeEvent(self, event)

    def _calculate_crop_size(self, scale_factor):
        """Return the (height, width) crop size for the given scale factor.

        The crop size is ``base_crop_size / scale_factor``, clamped to a minimum of 32.
        """
        crop_size = max(
            round(self.base_crop_size / round(scale_factor, 2)), 32
        )
        return (crop_size, crop_size)
