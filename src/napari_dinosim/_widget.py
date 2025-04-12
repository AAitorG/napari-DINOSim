import json
import os
import re
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
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QFrame,
    QHBoxLayout,
    QPushButton,
)

from .dinoSim_pipeline import DinoSim_pipeline
from .utils import gaussian_kernel, get_img_processing_f, torch_convolve

# Try to import SAM2
try:
    from .sam2_utils import SAM2Processor

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class CollapsibleSection(QWidget):
    """A collapsible section widget with a header and a content area.

    Parameters
    ----------
    title : str
        The title of the section.
    parent : QWidget, optional
        The parent widget.
    collapsed : bool, optional
        Whether the section is initially collapsed. Default is False.
    """

    def __init__(self, title, parent=None, collapsed=False):
        super().__init__(parent)

        # Create layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Create header
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(5, 5, 5, 5)

        # Toggle button
        self.toggle_button = QPushButton()
        self.toggle_button.setMaximumWidth(20)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        self.toggle_button.clicked.connect(self.toggle_content)

        # Section title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        # Add to header layout
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()

        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(20, 5, 5, 5)  # Indent content

        # Add line separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)

        # Add widgets to main layout
        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(line)
        self.main_layout.addWidget(self.content)

        # Set initial state
        self.update_toggle_button()
        if collapsed:
            self.content.setVisible(False)

    def toggle_content(self):
        """Toggle the visibility of the content area."""
        self.content.setVisible(self.toggle_button.isChecked())
        self.update_toggle_button()

    def update_toggle_button(self):
        """Update the toggle button text based on collapsed state."""
        if self.toggle_button.isChecked():
            self.toggle_button.setText("▼")  # Down arrow for expanded
        else:
            self.toggle_button.setText("►")  # Right arrow for collapsed

    def add_widget(self, widget):
        """Add a widget to the content area.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        """
        self.content_layout.addWidget(widget)

    @property
    def native(self):
        """Return self to maintain compatibility with magicgui."""
        return self


class DINOSim_widget(QWidget):
    """DINOSim napari widget for zero-shot image segmentation using DINO vision transformers.

    This widget provides a graphical interface for loading DINO models, selecting reference
    points in images, and generating segmentation masks based on visual similarity.

    Parameters
    ----------
    viewer : Viewer
        The napari viewer instance this widget will be attached to.

    Attributes
    ----------
    compute_device : torch.device
        The device (CPU/GPU) used for computation.
    model_dims : dict
        Dictionary mapping model sizes to their number of feature dimensions.
    base_crop_size : int
        Base crop size for scaling calculations.
    model : torch.nn.Module
        The loaded DINO vision transformer model.
    feat_dim : int
        Feature dimension of the current model.
    pipeline_engine : DinoSim_pipeline
        The processing pipeline for computing embeddings and similarities.
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
        scroll_area.setMinimumHeight(400)  # Set a minimum height for better UX

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
        self.model_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        # Base crop size for scaling calculations
        self.base_crop_size = 518
        self.model = None
        self.feat_dim = 0
        self.pipeline_engine = None
        self.upsample = "bilinear"  # bilinear, None
        self.resize_size = 518  # should be multiple of model patch_size
        kernel = gaussian_kernel(size=3, sigma=1)
        kernel = torch.tensor(
            kernel, dtype=torch.float32, device=self.compute_device
        )
        self.filter = lambda x: torch_convolve(x, kernel)  # gaussian filter
        self._points_layer: Optional[Points] = None
        self.loaded_img_layer: Optional[Image] = None
        # Store active workers to prevent premature garbage collection
        self._active_workers = []
        # Add flag for layer insertion
        self._is_inserting_layer = False
        # Add flag to prevent callback when programmatically changing scale factor
        self._is_programmatic_scale_change = False
        # Add flag to prevent callback when programmatically changing threshold
        self._is_programmatic_threshold_change = False

        # SAM2 related attributes
        self.has_sam2 = HAS_SAM2
        self.sam2_model_sizes = ["tiny", "small", "base", "large"]
        self.sam2_processor = None
        self.refined_mask = None

        # Show welcome dialog with instructions
        self._show_welcome_dialog()

        # Create all GUI elements and add them directly to the container
        self._create_gui()

        # Variables to store intermediate results
        self._references_coord = []
        self.predictions = None

        # Load default model after GUI setup
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

        Creates all UI elements and adds them directly to the container.
        """
        # Get the native layout of the main container
        container_layout = self.container.native.layout()
        if container_layout is None:
            # If no layout exists, create a QVBoxLayout
            container_layout = QVBoxLayout()
            self.container.native.setLayout(container_layout)
            # Set margins to avoid extra spacing around the container
            container_layout.setContentsMargins(0, 0, 0, 0)

        # Create title label and add its native widget to the layout
        title_label = Label(value="DINOSim")
        title_label.native.setStyleSheet(
            "font-weight: bold; font-size: 18px; qproperty-alignment: AlignCenter;"
        )
        container_layout.addWidget(title_label.native)

        # Create sections using the CollapsibleSection widget
        # Model section
        model_content = self._create_model_section()
        model_section = CollapsibleSection("Model Selection")
        model_section.add_widget(model_content.native)
        container_layout.addWidget(model_section)  # Add directly to layout

        # Processing section
        processing_content = self._create_processing_section()
        processing_section = CollapsibleSection("Settings")
        processing_section.add_widget(processing_content.native)
        container_layout.addWidget(
            processing_section
        )  # Add directly to layout

        # SAM2 section
        sam2_content = self._create_sam2_section()
        sam2_section = CollapsibleSection(
            "SAM2 Post-Processing", collapsed=True
        )
        sam2_section.add_widget(sam2_content.native)
        container_layout.addWidget(sam2_section)  # Add directly to layout

        # Add a stretch at the end to push content to the top
        container_layout.addStretch(1)

        # Set a maximum width for the whole container to prevent excessive stretching
        self.container.native.setMaximumWidth(450)

    def _create_model_section(self):
        """Create the model selection section of the GUI.

        Returns
        -------
        Container
            A widget container with model size selector, load button, and GPU status.
        """
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
        self._notification_checkbox = CheckBox(
            text=(
                "Available"
                if torch.cuda.is_available()
                or torch.backends.mps.is_available()
                else "Not Available"
            ),
            value=torch.cuda.is_available(),
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
        # Set a max width for the container
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
        """Create the image processing section of the GUI.

        Returns
        -------
        Container
            A widget container with reference controls, image selection,
            crop size selector, and threshold controls.
        """
        # Reference controls container
        ref_controls = self._create_reference_controls()

        image_layer_label = Label(
            value="Image to segment:", name="subsection_label"
        )
        self._image_layer_combo = create_widget(
            annotation="napari.layers.Image"
        )
        self._image_layer_combo.reset_choices()
        self._image_layer_combo.changed.connect(self._new_image_selected)

        # Add embedding status indicator
        self._emb_status_indicator = Label(value="  ")
        self._emb_status_indicator.native.setStyleSheet(
            "background-color: red; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
        )
        self._set_embedding_status(
            "unavailable"
        )  # Initial state is unavailable

        # Connect to layer name changes
        def _on_layer_name_changed(event):
            # Check if the layer still exists in the viewer
            if event.source in self._viewer.layers:
                current_value = self._image_layer_combo.value
                self._image_layer_combo.reset_choices()
                # Try to restore the selection if the renamed layer was the selected one
                if event.source == current_value:
                    self._image_layer_combo.value = event.source
                # If no layer was selected or the selected layer was removed,
                # default to the first available image layer if any
                elif (
                    not self._image_layer_combo.value
                    and self._image_layer_combo.choices
                ):
                    self._image_layer_combo.value = (
                        self._image_layer_combo.choices[0]
                    )

        # Connect to name changes for all existing layers
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.events.name.connect(_on_layer_name_changed)

        # Update connection in layer insertion handler
        def _connect_layer_name_change(layer):
            if isinstance(layer, Image):
                try:  # Disconnect first to avoid duplicates if re-added
                    layer.events.name.disconnect(_on_layer_name_changed)
                except TypeError:
                    pass  # Was not connected
                layer.events.name.connect(_on_layer_name_changed)

        self._viewer.layers.events.inserted.connect(
            lambda e: _connect_layer_name_change(e.value)
        )

        # Also handle removal to disconnect signal
        def _disconnect_layer_name_change(layer):
            if isinstance(layer, Image):
                try:
                    layer.events.name.disconnect(_on_layer_name_changed)
                except TypeError:
                    pass  # Was not connected

        self._viewer.layers.events.removed.connect(
            lambda e: _disconnect_layer_name_change(e.value)
        )

        image_layer_container = Container(
            widgets=[
                image_layer_label,
                self._image_layer_combo,
                self._emb_status_indicator,
            ],
            layout="horizontal",
            labels=False,
        )
        image_layer_container.native.setMaximumWidth(400)  # Set max width
        self._points_layer = None

        # Create a collapsible section for processing parameters
        params_section = CollapsibleSection(
            "Processing Parameters", collapsed=False
        )
        params_content = QWidget()
        params_layout = QVBoxLayout(params_content)
        params_layout.setContentsMargins(0, 0, 0, 0)

        # Add image selection to the top of parameters section
        params_layout.addWidget(image_layer_container.native)

        # Scale factor control
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
        crop_size_container.native.setMaximumWidth(400)  # Set max width
        params_layout.addWidget(crop_size_container.native)

        # Threshold control
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
        threshold_container.native.setMaximumWidth(400)  # Set max width
        params_layout.addWidget(threshold_container.native)

        # Add the parameters content to the section
        params_section.add_widget(params_content)

        # Create a collapsible section for embedding controls
        emb_section = CollapsibleSection("Embedding Controls", collapsed=True)
        emb_content = QWidget()
        emb_layout = QVBoxLayout(emb_content)
        emb_layout.setContentsMargins(0, 0, 0, 0)

        # Precomputation controls
        precompute_label = Label(
            value="Auto Precompute:", name="subsection_label"
        )
        self.auto_precompute_checkbox = CheckBox(
            value=True,
            text="",
            tooltip="Automatically precompute embeddings when image/crop size changes",
        )
        self.auto_precompute_checkbox.changed.connect(
            self._toggle_manual_precompute_button
        )

        # Create a horizontal container for the label and checkbox
        precompute_header = Container(
            widgets=[precompute_label, self.auto_precompute_checkbox],
            layout="horizontal",
            labels=False,
        )
        precompute_header.native.setMaximumWidth(400)  # Set max width
        emb_layout.addWidget(precompute_header.native)

        self.manual_precompute_btn = PushButton(
            text="Precompute Now",
            tooltip="Manually trigger embedding precomputation",
        )
        self.manual_precompute_btn.changed.connect(self._manual_precompute)
        self.manual_precompute_btn.enabled = (
            False  # Initially disabled since auto is on
        )
        emb_layout.addWidget(self.manual_precompute_btn.native)

        # Save/Load embeddings buttons
        self._save_emb_btn = PushButton(
            text="Save Embeddings",
            tooltip="Save precomputed embeddings to a file",
        )
        self._save_emb_btn.changed.connect(self._save_embeddings)

        self._load_emb_btn = PushButton(
            text="Load Embeddings",
            tooltip="Load embeddings from a file",
        )
        self._load_emb_btn.changed.connect(self._load_embeddings)

        emb_buttons = Container(
            widgets=[self._save_emb_btn, self._load_emb_btn],
            layout="horizontal",
            labels=False,
        )
        emb_buttons.native.setMaximumWidth(400)  # Set max width
        emb_layout.addWidget(emb_buttons.native)

        # Add the embedding content to the section
        emb_section.add_widget(emb_content)

        # Register event handlers
        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        # Reset button
        self._reset_btn = PushButton(
            text="Reset Default Settings",
            tooltip="Reset references and embeddings.",
        )
        self._reset_btn.changed.connect(self.reset_all)

        # Create containers for the collapsible sections
        params_container = Container(widgets=[params_section], labels=False)
        emb_container = Container(widgets=[emb_section], labels=False)

        return Container(
            widgets=[
                ref_controls,
                params_container,
                emb_container,
                self._reset_btn,
            ],
            labels=False,
        )

    def _create_sam2_section(self):
        """Create the SAM2 post-processing section of the GUI.

        This section is only available if SAM2 library is imported.

        Returns
        -------
        Container
            A widget container with SAM2 controls or a message if SAM2 is not available.
        """
        if not self.has_sam2:
            # SAM2 not available - show message
            sam2_unavailable_label = Label(
                value="SAM2 library not found. Please install it first.",
                name="info_label",
            )
            return Container(
                widgets=[sam2_unavailable_label],
                labels=False,
            )

        # SAM2 Enable checkbox
        enable_sam2_label = Label(
            value="Enable SAM2:", name="subsection_label"
        )
        self.enable_sam2_checkbox = CheckBox(
            value=False,
            text="",
            tooltip="Enable experimental SAM2 post-processing",
        )
        # Connect the checkbox to handler
        self.enable_sam2_checkbox.changed.connect(
            self._on_sam2_enabled_changed
        )

        # SAM2 status indicator
        self._sam2_status_indicator = Label(value="  ")
        self._sam2_status_indicator.native.setStyleSheet(
            "background-color: red; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
        )
        self._set_sam2_status("unavailable")  # Initial state is unavailable
        enable_sam2_container = Container(
            widgets=[
                enable_sam2_label,
                self.enable_sam2_checkbox,
                self._sam2_status_indicator,
            ],
            layout="horizontal",
            labels=False,
        )
        enable_sam2_container.native.setMaximumWidth(400)  # Set max width

        # SAM2 model size selector
        sam2_model_label = Label(
            value="SAM2 Model Size:", name="subsection_label"
        )
        self.sam2_model_selector = ComboBox(
            value="small",
            choices=self.sam2_model_sizes,
            tooltip="Select SAM2 model size. Larger models are more accurate but require more resources.",
        )
        # Connect the model selector to handler
        self.sam2_model_selector.changed.connect(
            self._on_sam2_model_size_changed
        )

        sam2_model_container = Container(
            widgets=[sam2_model_label, self.sam2_model_selector],
            layout="horizontal",
            labels=False,
        )
        sam2_model_container.native.setMaximumWidth(400)  # Set max width

        # Create SAM2 instances button and status indicator
        self.generate_sam2_instances_btn = PushButton(
            text="Generate Instances",
            tooltip="Generate instance segmentation using SAM2 model",
        )
        self.generate_sam2_instances_btn.changed.connect(
            self._generate_sam2_instances
        )

        sam2_button_container = Container(
            widgets=[self.generate_sam2_instances_btn],
            layout="horizontal",
            labels=False,
        )
        sam2_button_container.native.setMaximumWidth(400)  # Set max width

        return Container(
            widgets=[
                enable_sam2_container,
                sam2_model_container,
                sam2_button_container,
            ],
            labels=False,
        )

    def _toggle_manual_precompute_button(self):
        """Enable/disable manual precompute button based on checkbox state."""
        self.manual_precompute_btn.enabled = (
            not self.auto_precompute_checkbox.value
        )
        if self.pipeline_engine and not self.pipeline_engine.emb_precomputed:
            self._start_precomputation(
                finished_callback=self._update_reference_and_process
            )

    def _manual_precompute(self):
        """Handle manual precomputation button press."""
        self._start_precomputation(
            finished_callback=self._update_reference_and_process
        )

    def _create_reference_controls(self):
        """Create controls for managing reference points and embeddings.

        Returns
        -------
        Container
            A widget container with reference information display and save/load buttons.
        """
        # Create a container for reference controls
        ref_container = Container(layout="vertical", labels=False)

        # Create a collapsible subsection for reference points
        ref_subsection = CollapsibleSection(
            "Reference Information", collapsed=True
        )

        # Reference information content
        ref_content_widget = QWidget()
        ref_content_layout = QVBoxLayout(ref_content_widget)
        ref_content_layout.setContentsMargins(0, 0, 0, 0)

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
        ref_image_container.native.setMaximumWidth(400)  # Set max width

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
        ref_points_container.native.setMaximumWidth(400)  # Set max width

        # Add the containers to the reference content
        ref_content_layout.addWidget(ref_image_container.native)
        ref_content_layout.addWidget(ref_points_container.native)

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
        ref_buttons.native.setMaximumWidth(400)  # Set max width

        # Add reference buttons to the reference content
        ref_content_layout.addWidget(ref_buttons.native)

        # Add reference content to the subsection
        ref_subsection.add_widget(ref_content_widget)
        ref_container.append(ref_subsection)

        return ref_container

    def _save_reference(self):
        """Save the current reference to a file."""
        if (
            self.pipeline_engine is None
            or not self.pipeline_engine.exist_reference
        ):
            self._viewer.status = "No reference to save"
            return

        # Create default filename with pattern: reference_imagename.pt
        default_filename = "reference"
        if self._image_layer_combo.value is not None:
            # Add image name to filename
            image_name = self._image_layer_combo.value.name
            default_filename += f"_{image_name}"
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
        # Skip if this is triggered by layer insertion
        if hasattr(self, "_is_inserting_layer") and self._is_inserting_layer:
            return

        if self.pipeline_engine is None:
            self._set_embedding_status("unavailable")
            return
        self.pipeline_engine.delete_precomputed_embeddings()
        self._set_embedding_status("unavailable")

        # Only start precomputation if auto precompute is enabled
        if self.auto_precompute_checkbox.value:
            self._start_precomputation(finished_callback=self._get_dist_map)

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

    def _start_precomputation(self, finished_callback=None):
        """Centralized method for starting precomputation in a thread.

        Parameters
        ----------
        finished_callback : callable, optional
            Function to call when precomputation is complete
        """
        # Check if an image is selected
        if self._image_layer_combo.value is None:
            return

        # Update status indicator
        self._set_embedding_status("computing")

        # Update button text and style to show progress
        original_text = self.manual_precompute_btn.text
        original_style = self.manual_precompute_btn.native.styleSheet()
        self.manual_precompute_btn.text = "Precomputing..."
        self.manual_precompute_btn.native.setStyleSheet(
            "background-color: yellow; color: black;"
        )
        self.manual_precompute_btn.enabled = False

        def restore_button():
            """Restore button text, style and state after computation"""
            self.manual_precompute_btn.text = original_text
            self.manual_precompute_btn.native.setStyleSheet(original_style)
            self.manual_precompute_btn.enabled = (
                not self.auto_precompute_checkbox.value
            )

        # Update embedding status when complete
        def update_status_when_complete():
            if self.pipeline_engine and self.pipeline_engine.emb_precomputed:
                self._set_embedding_status("ready")
            else:
                self._set_embedding_status("unavailable")
            if finished_callback:
                finished_callback()

        # Create combined callback that restores button and runs user callback
        combined_callback = lambda: [
            restore_button(),
            update_status_when_complete(),
            self._sam2_precompute_masks(),
        ]

        worker = self.precompute_threaded()
        self._start_worker(
            worker,
            finished_callback=combined_callback,
            cleanup_callback=restore_button,  # Ensure button is restored even on error
        )
        return worker

    def _new_scale_factor_selected(self):
        """Handle scale factor change."""
        # Skip if this is a programmatic change
        if self._is_programmatic_scale_change:
            return

        self._reset_emb_and_ref()

        # Only start precomputation if auto precompute is enabled
        if self.auto_precompute_checkbox.value:
            self._start_precomputation(
                finished_callback=self._update_reference_and_process
            )

    def _check_existing_image_and_preprocess(self):
        """Check for existing image layers and preprocess if found."""
        image_found = False
        points_found = False
        for layer in self._viewer.layers:
            if not image_found and isinstance(layer, Image):
                self._image_layer_combo.value = layer

                # Update status based on whether embeddings are precomputed
                if (
                    self.pipeline_engine
                    and self.pipeline_engine.emb_precomputed
                ):
                    self._set_embedding_status("ready")
                else:
                    self._set_embedding_status("unavailable")

                # Only start precomputation if auto precompute is enabled
                if self.auto_precompute_checkbox.value:
                    self._start_precomputation()
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
        """Automatically precompute embeddings for the current image."""
        if self.pipeline_engine is not None:
            image_layer = self._image_layer_combo.value  # (n),h,w,(c)
            if image_layer is not None:
                image = self._get_nhwc_image(image_layer.data)
                assert image.shape[-1] in [
                    1,
                    3,
                    4,
                ], f"{image.shape[-1]} channels are not allowed, only 1, 3 or 4"
                if not self.pipeline_engine.emb_precomputed:
                    self.loaded_img_layer = self._image_layer_combo.value
                    # Calculate crop size from scale factor
                    crop_size = self._calculate_crop_size(
                        self.scale_factor_selector.value
                    )
                    self.pipeline_engine.pre_compute_embeddings(
                        image,
                        overlap=(0, 0),
                        padding=(0, 0),
                        crop_shape=(*crop_size, image.shape[-1]),
                        verbose=True,
                        batch_size=1,
                    )

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
            # Update status indicator
            self._set_embedding_status("unavailable")
            # Reset reference information labels
            self._ref_image_name.value = "None"
            self._ref_points_name.value = "None"

    def reset_all(self):
        """Reset references and embeddings."""
        if self.pipeline_engine is not None:
            # Set flag before changing threshold
            self._is_programmatic_threshold_change = True
            self._threshold_slider.value = 0.5
            self._is_programmatic_threshold_change = False

            # Set flag before changing scale factor
            self._is_programmatic_scale_change = True
            self.scale_factor_selector.value = 1.0
            self._is_programmatic_scale_change = False

            self._reset_emb_and_ref()

            # Only start precomputation if auto precompute is enabled
            if self.auto_precompute_checkbox.value:
                self._start_precomputation()

            # Reset SAM2-related variables
            self.refined_mask = None
            self._set_sam2_status("unavailable")

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

            # Reset the refined mask when we generate new predictions
            self.refined_mask = None

            # Check if SAM2 refinement is needed
            should_refine_sam2 = (
                self.has_sam2
                and self.enable_sam2_checkbox.value
                and self.sam2_processor is not None
            )

            if should_refine_sam2:
                # Start refinement in a separate thread
                worker = self._refine_with_sam2_threaded()
                self._set_sam2_status("computing")
                self._start_worker(
                    worker,
                    # The finished callback will handle updating predictions and thresholding
                    finished_callback=lambda: (
                        self._set_sam2_status("ready"),
                        self._threshold_im() if apply_threshold else None,
                    ),
                    cleanup_callback=lambda: (
                        # Ensure status is reset even on error
                        self._set_sam2_status(
                            "ready"
                            if self.sam2_processor.exist_predictions()
                            else "unavailable"
                        )
                    ),
                )
            else:
                # No SAM2 refinement needed, proceed with thresholding directly
                if apply_threshold:
                    self._threshold_im()

        except Exception as e:
            self._viewer.status = f"Error processing image: {str(e)}"

    @thread_worker
    def _refine_with_sam2_threaded(self):
        """Perform SAM2 refinement in a worker thread."""
        if (
            not self.sam2_processor.exist_predictions()
            and self._image_layer_combo.value is not None
        ):
            self._set_sam2_status("computing")
            image_data = self._get_nhwc_image(
                self._image_layer_combo.value.data
            )[0]
            self.sam2_processor.generate_sam_masks(image_data)
            self._set_sam2_status("ready")

        try:
            # Store original shape before processing
            original_shape = self.predictions.shape

            # Ensure prediction is a tensor for SAM2 refinement
            pred_for_refine = self.predictions
            if not isinstance(pred_for_refine, torch.Tensor):
                pred_for_refine = torch.tensor(
                    pred_for_refine, device=self.compute_device
                )

            # Apply refinement
            refined = self.sam2_processor.refine_prediction_with_sam_masks(
                pred_for_refine.squeeze()
            )

            # Ensure result is a tensor
            if not isinstance(refined, torch.Tensor):
                refined = torch.tensor(refined, device=self.compute_device)

            # Reshape to match original dimensions if needed
            if refined.ndim != len(original_shape):
                refined = refined.reshape(original_shape)

            # Store in separate attribute rather than overwriting predictions
            self.refined_mask = refined

            self._viewer.status = "SAM2 refinement complete."

        except Exception as e:
            self._viewer.status = f"Error during SAM2 refinement: {str(e)}"
            raise e  # Re-raise to trigger worker error handling

    def _threshold_im(self):
        # simple callback, otherwise numeric value is given as parameter
        # Skip if this is a programmatic change
        if self._is_programmatic_threshold_change:
            return
        self.threshold_im()

    def threshold_im(self, file_name=None):
        """Apply threshold to prediction map and display result.

        Parameters
        ----------
        file_name : str, optional
            Base name for the output mask layer. If None, uses input image name.
        """
        # Check if we have either predictions or refined mask
        if self.predictions is None and self.refined_mask is None:
            return

        # Determine which mask to use
        use_refined = (
            self.has_sam2
            and self.enable_sam2_checkbox.value
            and self.refined_mask is not None
        )

        if use_refined:
            # Use the SAM2 refined mask
            if isinstance(self.refined_mask, torch.Tensor):
                pred = self.refined_mask.cpu().numpy().copy()
            else:
                pred = self.refined_mask.copy()
        else:
            # Use the original DINO predictions
            if isinstance(self.predictions, torch.Tensor):
                pred = self.predictions.cpu().numpy().copy()
            else:
                pred = self.predictions.copy()

        # Ensure predictions are properly shaped for thresholding
        if pred.ndim > 2:
            pred = np.squeeze(pred)

        # Apply threshold - smaller values indicate greater similarity
        thresholded = pred < self._threshold_slider.value
        thresholded = (thresholded * 255).astype(np.uint8)

        # Get the name for the mask layer
        name = (
            self._image_layer_combo.value.name
            if file_name is None
            else file_name
        )
        name += "_mask"

        # Update or create the labels layer
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

                def after_precomputation():
                    self.pipeline_engine.set_reference_vector(
                        list_coords=self._references_coord, filter=self.filter
                    )
                    self._get_dist_map()

                # Only start precomputation if embeddings not already computed
                # and auto precompute is enabled
                if not self.pipeline_engine.emb_precomputed:
                    if self.auto_precompute_checkbox.value:
                        self._start_precomputation(
                            finished_callback=after_precomputation
                        )
                    else:
                        # If auto precompute is disabled, just show a status message
                        self._viewer.status = "Precomputation needed. Use the 'Precompute Now' button."
                else:
                    after_precomputation()

    def _load_model(self):
        self._image_layer_combo.reset_choices()
        try:
            # Clear CUDA cache before loading new model
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
        """Load the selected model based on the user's choice."""
        try:
            model_size = self.model_size_selector.value
            model_letter = model_size[0]

            if self.feat_dim != self.model_dims[model_size]:
                if self.model is not None:
                    self.model = None
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
                    f"Load New Model\n(Current Model: {model_size})"
                )

                if self.pipeline_engine is not None:
                    self.pipeline_engine = None

                interpolation = (
                    InterpolationMode.BILINEAR
                    if torch.backends.mps.is_available()
                    else InterpolationMode.BICUBIC
                )  # Bicubic is not implemented for MPS
                self.pipeline_engine = DinoSim_pipeline(
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
                # Set flag to prevent double precomputation
                self._is_inserting_layer = True
                # Reset choices before setting new value
                self._image_layer_combo.reset_choices()
                self._image_layer_combo.value = layer
                self._is_inserting_layer = False

                # Only precompute if needed and auto precompute is enabled
                if (
                    self.pipeline_engine
                    and self.pipeline_engine.emb_precomputed
                ):
                    if self.pipeline_engine.exist_reference:
                        self._get_dist_map()
                    else:
                        self._add_points_layer()
                elif self.auto_precompute_checkbox.value:
                    # Start precomputation with appropriate callback
                    if self.pipeline_engine:
                        if self.pipeline_engine.exist_reference:
                            self._start_precomputation(
                                finished_callback=self._get_dist_map
                            )
                        else:
                            self._start_precomputation(
                                finished_callback=self._add_points_layer
                            )
                    else:
                        self._start_precomputation()

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
            except (TypeError, RuntimeError):  # Add RuntimeError
                pass  # Handler was already disconnected or object deleted

            if self.pipeline_engine != None and self.loaded_img_layer == layer:
                self.pipeline_engine.delete_precomputed_embeddings()
                self.loaded_img_layer = None  # Set to None instead of ""
                self._set_embedding_status("unavailable")
            self._image_layer_combo.reset_choices()

        elif layer is self._points_layer:
            try:  # Wrap in try-except
                self._points_layer.events.data.disconnect(
                    self._update_reference_and_process
                )
            except (TypeError, RuntimeError):  # Add RuntimeError
                pass  # Already disconnected or object deleted
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

            # Clean up SAM2 resources
            if self.sam2_processor is not None:
                del self.sam2_processor
                self.refined_mask = None

            # Clear any remaining references
            self._active_workers.clear()

        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        finally:
            # Call QWidget's closeEvent instead of Container's
            QWidget.closeEvent(self, event)

    def _save_embeddings(self):
        """Save the precomputed embeddings to a file."""
        if (
            self.pipeline_engine is None
            or not self.pipeline_engine.emb_precomputed
        ):
            self._viewer.status = "No precomputed embeddings to save"
            return

        # Create default filename with pattern: embeddings_imagename_modelsize_scalingfactor.pt
        default_filename = "embeddings"
        if self._image_layer_combo.value is not None:
            # Add image name to filename
            image_name = self._image_layer_combo.value.name
            default_filename += f"_{image_name}"

        # Add model size and scaling factor
        model_size = self.model_size_selector.value
        scale_factor = self.scale_factor_selector.value
        default_filename += f"_{model_size}_x{scale_factor:.1f}.pt"

        filepath, _ = QFileDialog.getSaveFileName(
            None, "Save Embeddings", default_filename, "Embedding Files (*.pt)"
        )

        if filepath:
            if not filepath.endswith(".pt"):
                filepath += ".pt"
            try:
                self.pipeline_engine.save_embeddings(filepath)
                self._viewer.status = f"Embeddings saved to {filepath}"
            except Exception as e:
                self._viewer.status = f"Error saving embeddings: {str(e)}"

    def _load_embeddings(self):
        """Load embeddings from a file."""
        if self.pipeline_engine is None:
            self._viewer.status = "Model not loaded"
            return

        filepath, _ = QFileDialog.getOpenFileName(
            None, "Load Embeddings", "", "Embedding Files (*.pt)"
        )

        if filepath:
            try:
                self.pipeline_engine.load_embeddings(filepath)
                # Update status indicator
                self._set_embedding_status("ready")

                # Extract scale factor from filename and update the SpinBox.
                match = re.search(r"_x([0-9.]+)\.pt$", filepath)
                if match:
                    try:
                        # Set flag before changing scale factor
                        self._is_programmatic_scale_change = True
                        self.scale_factor_selector.value = float(
                            match.group(1)
                        )
                        self._is_programmatic_scale_change = False
                    except (
                        ValueError
                    ):  # Do not update the scale factor if the value does not match.
                        self._is_programmatic_scale_change = False
                        pass

                # Update references if they exist
                if (
                    self.pipeline_engine.exist_reference
                    and len(self._references_coord) > 0
                ):
                    self.pipeline_engine.set_reference_vector(
                        list_coords=self._references_coord, filter=self.filter
                    )

                self._get_dist_map()
                self._viewer.status = f"Embeddings loaded from {filepath}"
            except Exception as e:
                self._viewer.status = f"Error loading embeddings: {str(e)}"

    def _set_embedding_status(self, status):
        """Set the embedding status indicator color.

        Parameters
        ----------
        status : str
            One of: 'ready', 'computing', 'unavailable'
        """
        # Ensure indicator exists
        if (
            not hasattr(self, "_emb_status_indicator")
            or self._emb_status_indicator is None
        ):
            return

        if status == "ready":
            self._emb_status_indicator.native.setStyleSheet(
                "background-color: lightgreen; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._emb_status_indicator.tooltip = "Embeddings ready"
        elif status == "computing":
            self._emb_status_indicator.native.setStyleSheet(
                "background-color: yellow; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._emb_status_indicator.tooltip = "Computing embeddings..."
        else:  # 'unavailable'
            self._emb_status_indicator.native.setStyleSheet(
                "background-color: red; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._emb_status_indicator.tooltip = "Embeddings not available"

    def _calculate_crop_size(self, scale_factor):
        """Calculate crop size based on scale factor.

        Parameters
        ----------
        scale_factor : float
            The scale factor (e.g., 1.0, 2.0, 0.5)

        Returns
        -------
        tuple
            Crop dimensions (width, height)
        """
        # Calculate crop size - higher scale factor means smaller crop
        crop_size = round(self.base_crop_size / round(scale_factor, 2))
        # Ensure crop size is not too small
        crop_size = max(crop_size, 32)
        return (crop_size, crop_size)

    def _set_sam2_status(self, status):
        """Set the SAM2 status indicator color.

        Parameters
        ----------
        status : str
            One of: 'ready', 'computing', 'unavailable'
        """
        # Ensure indicator exists
        if (
            not hasattr(self, "_sam2_status_indicator")
            or self._sam2_status_indicator is None
        ):
            return

        if status == "ready":
            self._sam2_status_indicator.native.setStyleSheet(
                "background-color: lightgreen; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._sam2_status_indicator.tooltip = "SAM2 masks ready"
        elif status == "computing":
            self._sam2_status_indicator.native.setStyleSheet(
                "background-color: yellow; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._sam2_status_indicator.tooltip = "Computing SAM2 masks..."
        else:  # 'unavailable'
            self._sam2_status_indicator.native.setStyleSheet(
                "background-color: red; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._sam2_status_indicator.tooltip = "SAM2 masks not available"

    def _generate_sam2_instances(self):
        """Generate instance segmentation using SAM2."""
        if not self.has_sam2:
            self._viewer.status = (
                "SAM2 library not available. Please install it first."
            )
            return

        if not self.enable_sam2_checkbox.value:
            self._viewer.status = (
                "SAM2 is not enabled. Please enable it first."
            )
            return

        # Check if we have a processor and an image selected
        if self.sam2_processor is None:
            self._viewer.status = "Initializing SAM2 processor..."
            self._on_sam2_enabled_changed()  # Initialize processor if needed
            return

        if self._image_layer_combo.value is None:
            self._viewer.status = "No image selected for processing"
            return

        # Check if we have a prediction
        if self.predictions is None:
            self._viewer.status = "No segmentation prediction available. Select reference points first."
            return

        # Update status indicator to show computation in progress
        self._set_sam2_status("computing")
        self.generate_sam2_instances_btn.text = "Computing..."
        self.generate_sam2_instances_btn.enabled = False

        try:
            # Get image data
            image_layer = self._image_layer_combo.value
            image_data = self._get_nhwc_image(image_layer.data)[0]

            # Generate SAM2 masks for the image if not already done
            if not self.sam2_processor.exist_predictions():
                self.sam2_processor.generate_sam_masks(image_data)

            # Get threshold value and prepare prediction tensor
            threshold_value = self._threshold_slider.value
            pred_obj_white = False  # Assuming dark objects are foreground

            # Convert prediction to tensor - ensure we have the correct dimensionality
            # Use the original DINO predictions for instance generation
            pred_tensor = torch.tensor(
                np.squeeze(self.predictions),
                dtype=torch.float32,
                device=self.compute_device,
            )

            # Get instance segmentation using SAM2
            instances = (
                self.sam2_processor.get_refined_instances_with_sam_prediction(
                    pred_tensor,
                    pred_obj_white=pred_obj_white,
                    threshold=threshold_value,
                )
            )

            # Convert to numpy for display
            instances_np = instances.cpu().numpy()

            # Add to viewer as labels layer
            name = f"{image_layer.name}_instances"
            if name in self._viewer.layers:
                self._viewer.layers[name].data = instances_np
            else:
                self._viewer.add_labels(instances_np, name=name)

            self._set_sam2_status("ready")
            self._viewer.status = "SAM2 instance segmentation complete"

        except Exception as e:
            self._viewer.status = f"Error generating SAM2 instances: {str(e)}"
            self._set_sam2_status("unavailable")
        finally:
            # Always restore button state
            self.generate_sam2_instances_btn.text = "Generate Instances"
            self.generate_sam2_instances_btn.enabled = True

    def _on_sam2_enabled_changed(self):
        """Handle changes to the SAM2 enable checkbox."""
        if not self.has_sam2:
            return

        if self.enable_sam2_checkbox.value:
            if self.sam2_processor is None:
                # Enable SAM2 - initialize processor if it doesn't exist
                worker = self.init_sam2_processor()
                self._start_worker(worker)
        else:
            if self.sam2_processor is not None:
                self.sam2_processor._delete_predictions()
                self._set_sam2_status("unavailable")

    def _on_sam2_model_size_changed(self):
        """Handle changes to the SAM2 model size selector."""
        if not self.has_sam2 or not self.enable_sam2_checkbox.value:
            return

        # Update the SAM2 processor with the new model size
        worker = self._update_sam2_processor()
        self._start_worker(worker)

    @thread_worker
    def init_sam2_processor(self):
        """Initialize the SAM2 processor with the selected model size."""
        try:
            self.sam2_processor = SAM2Processor(device=self.compute_device)
            model_size = self.sam2_model_selector.value
            self.sam2_processor.load_model(model_type=model_size)

            # Generate masks if an image is loaded
            if self._image_layer_combo.value is not None:
                self._set_sam2_status("computing")
                image_data = self._get_nhwc_image(
                    self._image_layer_combo.value.data
                )[0]
                self.sam2_processor.generate_sam_masks(image_data)
                self._set_sam2_status("ready")

                # Re-process the current predictions with SAM2
                if self.predictions is not None:
                    self._get_dist_map()
            else:
                self._set_sam2_status("ready")

            self._viewer.status = (
                f"SAM2 {model_size} model initialized successfully"
            )
        except Exception as e:
            self._set_sam2_status("unavailable")
            self.enable_sam2_checkbox.value = False
            self._viewer.status = f"Error initializing SAM2: {str(e)}"
            raise e  # Re-raise to propagate to the worker error handler

    def _sam2_precompute_masks(self):
        """Precompute SAM2 masks for the current image."""
        # Update SAM2 masks for new image if SAM2 is enabled
        if (
            self.has_sam2
            and self.enable_sam2_checkbox.value
            and self.sam2_processor is not None
        ):
            if self._image_layer_combo.value is not None:
                self._set_sam2_status("computing")
                image_data = self._get_nhwc_image(
                    self._image_layer_combo.value.data
                )[0]
                self.sam2_processor.generate_sam_masks(image_data)
                self._set_sam2_status("ready")
                # Reset the refined mask when we generate new SAM2 masks
                self.refined_mask = None

    @thread_worker
    def _update_sam2_processor(self):
        """Update the SAM2 processor with the current settings."""
        if not self.has_sam2 or self.sam2_processor is None:
            return

        model_size = self.sam2_model_selector.value

        # Only reload if the model size has changed
        if self.sam2_processor.model_type != model_size:
            self.sam2_processor.load_model(model_type=model_size)
            if self._image_layer_combo.value is not None:
                self._set_sam2_status("computing")
                image_data = self._get_nhwc_image(
                    self._image_layer_combo.value.data
                )[0]
                self.sam2_processor.generate_sam_masks(image_data)
                self._set_sam2_status("ready")
                # Reset the refined mask when we reload the model
                self.refined_mask = None

                # Re-process if we have predictions
                if self.predictions is not None:
                    self._get_dist_map()

    # Add compatibility methods for napari
    @property
    def native(self):
        """Return the native Qt widget"""
        return self

    def _get_widget(self):
        """Return the widget itself for napari"""
        return self
