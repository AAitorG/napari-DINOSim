import re

from magicgui.widgets import (
    CheckBox,
    Label,
    PushButton,
)
from napari.qt import thread_worker
from qtpy.QtWidgets import QFileDialog

from .utils import (
    ensure_valid_dtype,
    get_nhwc_image,
)


class EmbeddingManager:
    """Helper class to manage embedding precomputation, saving, and loading."""

    def __init__(self, parent):
        self.parent = parent

        # UI elements related to embeddings
        self._emb_status_indicator = Label(value="  ")
        self._emb_status_indicator.native.setStyleSheet(
            "background-color: red; border-radius: 8px; min-width: 16px; min-height: 16px; max-width: 16px; max-height: 16px;"
        )
        self.set_embedding_status("unavailable")

        self.auto_precompute_checkbox = CheckBox(
            value=True,
            text="",
            tooltip="Automatically precompute embeddings when image/crop size changes",
        )
        self.auto_precompute_checkbox.changed.connect(
            self.toggle_manual_precompute_button
        )

        self.manual_precompute_btn = PushButton(
            text="Precompute Now",
            tooltip="Manually trigger embedding precomputation",
        )
        self.manual_precompute_btn.changed.connect(self.manual_precompute)
        self.manual_precompute_btn.enabled = False

        self.save_emb_btn = PushButton(
            text="Save Embeddings",
            tooltip="Save precomputed embeddings to a file",
        )
        self.save_emb_btn.changed.connect(self.save_embeddings)

        self.load_emb_btn = PushButton(
            text="Load Embeddings",
            tooltip="Load embeddings from a file",
        )
        self.load_emb_btn.changed.connect(self.load_embeddings)

    def set_embedding_status(self, status):
        """Set the embedding status indicator color."""
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

    def toggle_manual_precompute_button(self):
        """Enable/disable manual precompute button based on checkbox state."""
        self.manual_precompute_btn.enabled = (
            not self.auto_precompute_checkbox.value
        )
        if (
            self.parent.pipeline_engine
            and not self.parent.pipeline_engine.emb_precomputed
        ):
            self.start_precomputation(
                finished_callback=self.parent._update_reference_and_process
            )

    def manual_precompute(self):
        """Handle manual precomputation button press."""
        self.start_precomputation(
            finished_callback=self.parent._update_reference_and_process
        )

    def start_precomputation(self, finished_callback=None):
        """Centralized method for starting precomputation in a thread."""
        if self.parent._image_layer_combo.value is None:
            return

        self.set_embedding_status("computing")

        original_text = self.manual_precompute_btn.text
        original_style = self.manual_precompute_btn.native.styleSheet()
        self.manual_precompute_btn.text = "Precomputing..."
        self.manual_precompute_btn.native.setStyleSheet(
            "background-color: yellow; color: black;"
        )
        self.manual_precompute_btn.enabled = False

        def restore_button():
            self.manual_precompute_btn.text = original_text
            self.manual_precompute_btn.native.setStyleSheet(original_style)
            self.manual_precompute_btn.enabled = (
                not self.auto_precompute_checkbox.value
            )

        def update_status_when_complete():
            if (
                self.parent.pipeline_engine
                and self.parent.pipeline_engine.emb_precomputed
            ):
                self.set_embedding_status("ready")
            else:
                self.set_embedding_status("unavailable")
                restore_button()
            if finished_callback:
                finished_callback()

        combined_callback = lambda: [
            restore_button(),
            update_status_when_complete(),
        ]

        worker = self.precompute_threaded()
        self.parent._start_worker(
            worker,
            finished_callback=combined_callback,
            cleanup_callback=restore_button,
        )
        return worker

    @thread_worker()
    def precompute_threaded(self):
        """Worker thread that calls auto_precompute() in the background."""
        self.auto_precompute()

    def auto_precompute(self):
        """Automatically precompute embeddings for the current image."""
        if self.parent.pipeline_engine is not None:
            image_layer = self.parent._image_layer_combo.value
            if image_layer is not None:
                image = get_nhwc_image(image_layer.data)
                assert image.shape[-1] in [
                    1,
                    3,
                    4,
                ], f"{image.shape[-1]} channels are not allowed, only 1, 3 or 4"
                if not self.parent.pipeline_engine.emb_precomputed:
                    self.parent.loaded_img_layer = (
                        self.parent._image_layer_combo.value
                    )
                    crop_size = self.parent._calculate_crop_size(
                        self.parent.scale_factor_selector.value
                    )
                    image = ensure_valid_dtype(image)
                    self.parent.pipeline_engine.pre_compute_embeddings(
                        image,
                        overlap=(0, 0),
                        padding=(0, 0),
                        crop_shape=(*crop_size, image.shape[-1]),
                        verbose=True,
                        batch_size=1,
                    )

    def save_embeddings(self):
        """Save the precomputed embeddings to a file."""
        if (
            self.parent.pipeline_engine is None
            or not self.parent.pipeline_engine.emb_precomputed
        ):
            self.parent._viewer.status = "No precomputed embeddings to save"
            return

        default_filename = "embeddings"
        if self.parent._image_layer_combo.value is not None:
            image_name = self.parent._image_layer_combo.value.name
            default_filename += f"_{image_name}"

        model_size = self.parent.model_size_selector.value
        scale_factor = self.parent.scale_factor_selector.value
        default_filename += f"_{model_size}_x{scale_factor:.1f}.pt"

        filepath, _ = QFileDialog.getSaveFileName(
            None, "Save Embeddings", default_filename, "Embedding Files (*.pt)"
        )

        if filepath:
            if not filepath.endswith(".pt"):
                filepath += ".pt"
            try:
                self.parent.pipeline_engine.save_embeddings(filepath)
                self.parent._viewer.status = f"Embeddings saved to {filepath}"
            except Exception as e:
                self.parent._viewer.status = (
                    f"Error saving embeddings: {str(e)}"
                )

    def load_embeddings(self):
        """Load embeddings from a file."""
        if self.parent.pipeline_engine is None:
            self.parent._viewer.status = "Model not loaded"
            return

        filepath, _ = QFileDialog.getOpenFileName(
            None, "Load Embeddings", "", "Embedding Files (*.pt)"
        )

        if filepath:
            try:
                self.parent.pipeline_engine.load_embeddings(filepath)
                self.set_embedding_status("ready")

                match = re.search(r"_x([0-9.]+)\.pt$", filepath)
                if match:
                    try:
                        self.parent._is_programmatic_scale_change = True
                        self.parent.scale_factor_selector.value = float(
                            match.group(1)
                        )
                        self.parent._is_programmatic_scale_change = False
                    except ValueError:
                        self.parent._is_programmatic_scale_change = False

                if (
                    self.parent.pipeline_engine.exist_reference
                    and len(self.parent._references_coord) > 0
                ):
                    self.parent.pipeline_engine.set_reference_vector(
                        list_coords=self.parent._references_coord,
                        filter=self.parent.filter,
                    )

                self.parent._get_dist_map()
                self.parent._viewer.status = (
                    f"Embeddings loaded from {filepath}"
                )
            except Exception as e:
                self.parent._viewer.status = (
                    f"Error loading embeddings: {str(e)}"
                )
