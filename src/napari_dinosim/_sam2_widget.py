import numpy as np
import torch
from napari.qt import thread_worker
from qtpy.QtWidgets import QFileDialog

from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    PushButton,
)

# Try to import SAM2
try:
    from .utils import SAM2Processor

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False


class SAM2WidgetHelper:
    """Helper class to manage SAM2-related UI and logic for DINOSim_widget."""

    def __init__(self, parent):
        self.parent = parent
        self.has_sam2 = HAS_SAM2

        # SAM2 related attributes (moved from parent)
        self.sam2_processor = None
        self.refined_mask = None

    def create_sam2_section(self):
        """Create the SAM2 post-processing section of the GUI."""
        if not self.has_sam2:
            # SAM2 not available - show message
            sam2_unavailable_label = Label(
                value="SAM2 library not installed. \nPlease check the documentation.",
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
            tooltip="Enable SAM2 post-processing with precomputed masks",
        )
        # Connect the checkbox to handler
        self.enable_sam2_checkbox.changed.connect(self.on_sam2_enabled_changed)

        # SAM2 status indicator
        self._sam2_status_indicator = Label(value="  ")
        self._sam2_status_indicator.native.setStyleSheet(
            "background-color: red; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
        )
        self.set_sam2_status("unavailable")  # Initial state is unavailable

        # Put status indicator to the left of the checkbox
        enable_sam2_container = Container(
            widgets=[
                enable_sam2_label,
                self._sam2_status_indicator,
                self.enable_sam2_checkbox,
            ],
            layout="horizontal",
            labels=False,
        )
        enable_sam2_container.native.setMaximumWidth(400)  # Set max width

        # Load SAM2 masks button
        self.load_sam2_masks_btn = PushButton(
            text="Load SAM2 Masks",
            tooltip="Load precomputed SAM2 masks from a file",
        )
        self.load_sam2_masks_btn.changed.connect(self.load_sam2_masks)

        # Create SAM2 instances button
        self.generate_sam2_instances_btn = PushButton(
            text="Generate Instances",
            tooltip="Generate instance segmentation using loaded SAM2 masks",
        )
        self.generate_sam2_instances_btn.changed.connect(
            self.generate_sam2_instances
        )

        # Put Load and Generate buttons next to each other
        sam2_button_container = Container(
            widgets=[
                self.load_sam2_masks_btn,
                self.generate_sam2_instances_btn,
            ],
            layout="horizontal",
            labels=False,
        )
        sam2_button_container.native.setMaximumWidth(400)  # Set max width

        return Container(
            widgets=[
                enable_sam2_container,
                sam2_button_container,
            ],
            labels=False,
        )

    def set_sam2_status(self, status):
        """Set the SAM2 status indicator color."""
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
            if hasattr(self, "load_sam2_masks_btn"):
                self.load_sam2_masks_btn.native.setStyleSheet(
                    "background-color: lightgreen; color: black;"
                )
        elif status == "computing":
            self._sam2_status_indicator.native.setStyleSheet(
                "background-color: yellow; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._sam2_status_indicator.tooltip = "Computing SAM2 masks..."
            if hasattr(self, "load_sam2_masks_btn"):
                self.load_sam2_masks_btn.native.setStyleSheet(
                    "background-color: yellow; color: black;"
                )
        else:  # 'unavailable'
            self._sam2_status_indicator.native.setStyleSheet(
                "background-color: red; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
            )
            self._sam2_status_indicator.tooltip = "SAM2 masks not available"
            if hasattr(self, "load_sam2_masks_btn"):
                self.load_sam2_masks_btn.native.setStyleSheet("")

    def on_sam2_enabled_changed(self):
        """Handle changes to the SAM2 enable checkbox."""
        if not self.has_sam2:
            return

        if self.enable_sam2_checkbox.value:
            if self.sam2_processor is None:
                worker = self.init_sam2_processor()
                self.parent._start_worker(worker)
            else:
                has_predictions = self.sam2_processor.exist_predictions()
                self.set_sam2_status(
                    "ready" if has_predictions else "unavailable"
                )

                if self.parent.predictions is not None and has_predictions:
                    worker = self.refine_with_sam2_threaded()
                    self.parent._start_worker(
                        worker,
                        finished_callback=lambda: (
                            self.set_sam2_status("ready"),
                            self.parent._threshold_im(),
                        ),
                    )
        else:
            self.set_sam2_status("unavailable")
            if self.parent.predictions is not None:
                self.parent._threshold_im()

    @thread_worker
    def init_sam2_processor(self):
        """Initialize the SAM2 processor for precomputed masks only."""
        try:
            self.sam2_processor = SAM2Processor(
                device=self.parent.sam2_compute_device
            )
            self.set_sam2_status("unavailable")
            self.parent._viewer.status = "SAM2 processor initialized for precomputed masks. Please load masks."
        except Exception as e:
            self.set_sam2_status("unavailable")
            self.enable_sam2_checkbox.value = False
            self.parent._viewer.status = f"Error initializing SAM2: {str(e)}"
            raise e

    def load_sam2_masks(self):
        """Load precomputed SAM2 masks from a file."""
        if not self.has_sam2:
            self.parent._viewer.status = (
                "SAM2 library not installed. \nPlease check the documentation."
            )
            return

        if self.sam2_processor is None:
            worker = self.init_sam2_processor()
            self.parent._start_worker(
                worker, finished_callback=self.show_load_masks_dialog
            )
        else:
            self.show_load_masks_dialog()

    def show_load_masks_dialog(self):
        """Show file dialog to load SAM2 masks."""
        filepath, _ = QFileDialog.getOpenFileName(
            None, "Load SAM2 Masks", "", "SAM2 Mask Files (*.pt)"
        )

        if filepath:
            try:
                self.set_sam2_status("computing")
                self.sam2_processor.load_masks(filepath)
                self.set_sam2_status("ready")
                self.parent._viewer.status = (
                    f"SAM2 masks loaded from {filepath}"
                )

                if self.parent.predictions is not None:
                    worker = self.refine_with_sam2_threaded()
                    self.parent._start_worker(
                        worker,
                        finished_callback=lambda: (
                            self.set_sam2_status("ready"),
                            self.parent._threshold_im(),
                        ),
                    )
            except Exception as e:
                self.parent._viewer.status = (
                    f"Error loading SAM2 masks: {str(e)}"
                )
                self.set_sam2_status("unavailable")

    def generate_sam2_instances(self):
        """Generate instance segmentation using loaded SAM2 masks."""
        if not self.has_sam2:
            self.parent._viewer.status = (
                "SAM2 library not installed. \nPlease check the documentation."
            )
            return

        if not self.enable_sam2_checkbox.value:
            self.parent._viewer.status = (
                "SAM2 is not enabled. Please enable it first."
            )
            return

        if self.sam2_processor is None:
            self.parent._viewer.status = "Initializing SAM2 processor..."
            self.on_sam2_enabled_changed()
            return

        if not self.sam2_processor.exist_predictions():
            self.parent._viewer.status = (
                "No SAM2 masks loaded. Please load masks first."
            )
            return

        if self.parent._image_layer_combo.value is None:
            self.parent._viewer.status = "No image selected for processing"
            return

        if self.parent.predictions is None:
            self.parent._viewer.status = "No segmentation prediction available. Select reference points first."
            return

        self.set_sam2_status("computing")
        self.generate_sam2_instances_btn.text = "Computing..."
        self.generate_sam2_instances_btn.enabled = False

        try:
            image_layer = self.parent._image_layer_combo.value
            threshold_value = self.parent._threshold_slider.value
            pred_obj_white = False

            pred_tensor = torch.tensor(
                np.squeeze(self.parent.predictions),
                dtype=torch.float32,
                device=self.parent.sam2_compute_device,
            )

            instances_np = (
                self.sam2_processor.get_refined_instances_with_sam_prediction(
                    pred_tensor,
                    pred_obj_white=pred_obj_white,
                    threshold=threshold_value,
                )
            )

            name = f"{image_layer.name}_instances"
            if name in self.parent._viewer.layers:
                self.parent._viewer.layers[name].data = instances_np
            else:
                self.parent._viewer.add_labels(instances_np, name=name)

            self.set_sam2_status("ready")
            self.parent._viewer.status = "SAM2 instance segmentation complete"

        except Exception as e:
            self.parent._viewer.status = (
                f"Error generating SAM2 instances: {str(e)}"
            )
            self.set_sam2_status("unavailable")
        finally:
            self.generate_sam2_instances_btn.text = "Generate Instances"
            self.generate_sam2_instances_btn.enabled = True

    @thread_worker
    def refine_with_sam2_threaded(self):
        """Apply SAM2 refinement to the current predictions."""
        if (
            self.sam2_processor is None
            or not self.sam2_processor.exist_predictions()
        ):
            self.parent._viewer.status = (
                "No SAM2 masks loaded. Please load masks first."
            )
            return

        try:
            if isinstance(self.parent.predictions, torch.Tensor):
                pred_for_refine = self.parent.predictions.clone()
            else:
                pred_for_refine = torch.tensor(
                    self.parent.predictions,
                    dtype=torch.float32,
                    device=self.parent.sam2_compute_device,
                )

            refined = self.sam2_processor.refine_prediction_with_sam_masks(
                pred_for_refine.squeeze()
            )

            self.refined_mask = refined
            self.parent._viewer.status = "SAM2 refinement complete."

        except Exception as e:
            self.parent._viewer.status = (
                f"Error during SAM2 refinement: {str(e)}"
            )
            raise e
