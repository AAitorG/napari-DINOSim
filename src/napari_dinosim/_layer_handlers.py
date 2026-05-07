from napari.layers import Image, Points
import logging

logger = logging.getLogger(__name__)


class LayerEventHandler:
    """Helper class to handle napari layer events for DINOSim_widget."""

    def __init__(self, parent):
        self.parent = parent

    def on_layer_inserted(self, event):
        """Handle a new layer being added to the viewer.

        For Points layers: switches the active points layer and reconnects the data-change
        callback so reference updates follow the latest points layer.
        For Image layers: resets the image combo to the new layer and, if a model is loaded,
        triggers precomputation or distance-map recomputation as appropriate.
        """
        layer = event.value
        try:
            if (
                isinstance(layer, Points)
                and layer is not self.parent._points_layer
            ):
                if self.parent._points_layer is not None:
                    try:
                        self.parent._points_layer.events.data.disconnect(
                            self.parent._update_reference_and_process
                        )
                    except (TypeError, RuntimeError):
                        pass

                layer.mode = "add"
                self.parent._points_layer = layer
                self.parent._points_layer.events.data.connect(
                    self.parent._update_reference_and_process
                )
                self.parent._update_reference_and_process()

            elif isinstance(layer, Image):
                self.parent._is_inserting_layer = True
                self.parent._image_layer_combo.reset_choices()
                self.parent._image_layer_combo.value = layer
                self.parent._is_inserting_layer = False

                if (
                    self.parent.pipeline_engine
                    and self.parent.pipeline_engine.emb_precomputed
                ):
                    if self.parent.pipeline_engine.exist_reference:
                        self.parent._get_dist_map()
                    else:
                        self.parent._add_points_layer()
                elif (
                    self.parent.embedding_manager.auto_precompute_checkbox.value
                ):
                    self.parent.sam2_helper.refined_mask = None

                    if self.parent.pipeline_engine:
                        if self.parent.pipeline_engine.exist_reference:
                            self.parent.embedding_manager.start_precomputation(
                                finished_callback=self.parent._get_dist_map
                            )
                        else:
                            self.parent.embedding_manager.start_precomputation(
                                finished_callback=self.parent._add_points_layer
                            )
                    else:
                        self.parent.embedding_manager.start_precomputation()
        except Exception as e:
            logger.error(f"Error handling layer insertion: {e}", exc_info=True)
            self.parent._viewer.status = f"Error: {str(e)}"

    def on_layer_removed(self, event):
        """Handle a layer being removed from the viewer.

        For Image layers: invalidates precomputed embeddings if the removed layer was the
        one used for embedding. For the active Points layer: disconnects data events and
        clears the stored reference on the pipeline.
        """
        layer = event.value

        if isinstance(layer, Image):
            try:
                layer.events.name.disconnect()
            except (TypeError, RuntimeError):
                pass

            if (
                self.parent.pipeline_engine is not None
                and self.parent.loaded_img_layer == layer
            ):
                self.parent.pipeline_engine.delete_precomputed_embeddings()
                self.parent.loaded_img_layer = None
                self.parent.embedding_manager.set_embedding_status(
                    "unavailable"
                )
            self.parent._image_layer_combo.reset_choices()

        elif layer is self.parent._points_layer:
            try:
                self.parent._points_layer.events.data.disconnect(
                    self.parent._update_reference_and_process
                )
            except (TypeError, RuntimeError):
                pass
            self.parent._points_layer = None
            if self.parent.pipeline_engine is not None:
                self.parent.pipeline_engine.delete_references()

    def on_layer_name_changed(self, event):
        """Handle layer name changes to keep the combo box in sync."""
        if event.source in self.parent._viewer.layers:
            current_value = self.parent._image_layer_combo.value
            self.parent._image_layer_combo.reset_choices()
            if event.source == current_value:
                self.parent._image_layer_combo.value = event.source
            elif (
                not self.parent._image_layer_combo.value
                and self.parent._image_layer_combo.choices
            ):
                self.parent._image_layer_combo.value = (
                    self.parent._image_layer_combo.choices[0]
                )

    def connect_layer_name_change(self, layer):
        """Connect the name-change event of an Image layer to the combo-box sync handler."""
        if isinstance(layer, Image):
            try:
                layer.events.name.disconnect(self.on_layer_name_changed)
            except TypeError:
                pass
            layer.events.name.connect(self.on_layer_name_changed)

    def disconnect_layer_name_change(self, layer):
        """Disconnect the name-change event of an Image layer from the combo-box sync handler."""
        if isinstance(layer, Image):
            try:
                layer.events.name.disconnect(self.on_layer_name_changed)
            except TypeError:
                pass
