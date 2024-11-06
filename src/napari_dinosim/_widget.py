from typing import Optional

from magicgui.widgets import CheckBox, Container, create_widget, ComboBox, PushButton, Label, ProgressBar, ToolBar
from scipy.ndimage import convolve, median_filter

from .utils import get_transforms, gaussian_kernel
from .dinoSim_pipeline  import DinoSim_pipeline

import os
from torch import hub, cuda, device
import numpy as np
from napari.qt import thread_worker
from napari.qt.threading import create_worker
from napari.layers import Image, Points

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DinoSim_widget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.model_dims = {'small': 384, 'base': 768, 'large': 1024, 'giant': 1536}
        self.crop_sizes = {'512x512':(512,512), '384x384':(384,384), '256x256':(256,256)}
        self.model = None
        self.feat_dim = 0
        self.pipeline_engine = None
        self.use_references_coord = True
        self.use_euclidean_dist = True
        self.upsample = 1  # 0:NN, 1:bilinear, None
        self.resize_size = 518  # should be multiple of model patch_size
        self.trfm = get_transforms(resize_size=self.resize_size)
        kernel = gaussian_kernel(size=3, sigma=1)
        self.post_processings = {'None':None, 'gaussian':lambda x: convolve(x[...,0], kernel), 'median':lambda x: median_filter(x, size=3)}
        self._points_layer: Optional["napari.layers.Points"] = None
        self.loaded_img_layer: Optional["napari.layers.Image"] = None

        # GUI elements
        self._create_gui()

        # Variables to store intermediate results
        self._references_coord = []
        self.predictions = None
        self._load_model()

    def _create_gui(self):
        """Create and organize the GUI components."""
        model_section = self._create_model_section()
        processing_section = self._create_processing_section()
        batch_processing_section = self._create_batch_processing_section()

        # Organize the main container
        self.extend(
            [
                model_section,
                ToolBar(),
                processing_section,
                ToolBar(),
                batch_processing_section
            ]
        )

    def _create_model_section(self):
        """Create the model selection section of the GUI."""
        model_section_label = Label(value="Model Selection", name="section_label")
        model_section_label.native.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        model_size_label = Label(value="Model Size:", name="subsection_label")
        self.model_size_selector = ComboBox(
            value='small',
            choices=list(self.model_dims.keys()),
            tooltip="Select the model size (s=small, b=base, l=large, g=giant). Larger models may be more accurate but require more resources."
        )
        model_size_container = Container(widgets=[model_size_label, self.model_size_selector], layout="horizontal", labels=False)
        
        self._load_model_btn = PushButton(
            text="Load Model",
            tooltip="Download (if necessary) and load selected model.",
        )
        self._load_model_btn.changed.connect(self._load_model)
        self._load_model_btn.native.setStyleSheet("background-color: red; color: black;")

        gpu_label = Label(value="GPU Status:", name="subsection_label")
        self._notification_checkbox = CheckBox(
            text="Available" if cuda.is_available() else "Not Available",
            value=cuda.is_available(),
        )
        self._notification_checkbox.enabled = False
        self._notification_checkbox.native.setStyleSheet("QCheckBox::indicator { width: 20px; height: 20px; }")
        gpu_container = Container(widgets=[gpu_label, self._notification_checkbox], layout="horizontal", labels=False)

        return Container(widgets=[
            model_section_label,
            model_size_container,
            self._load_model_btn,
            gpu_container
        ], labels=False)

    def _create_processing_section(self):
        """Create the image processing section of the GUI."""
        image_section_label = Label(value="Image Processing", name="section_label")
        image_section_label.native.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        image_layer_label = Label(value="Image Layer:", name="subsection_label")
        self._image_layer_combo = create_widget(annotation="napari.layers.Image")
        self._image_layer_combo.native.setStyleSheet("QComboBox { max-width: 200px; }")
        self._image_layer_combo.reset_choices()
        self._image_layer_combo.changed.connect(self._new_image_selected)
        image_layer_container = Container(widgets=[image_layer_label, self._image_layer_combo], layout="horizontal", labels=False)
        self._points_layer = None

        crop_size_label = Label(value="Crop Size:", name="subsection_label")
        self.crop_size_selector = ComboBox(
            value='512x512',
            choices=list(self.crop_sizes.keys()),
            tooltip="Select the model size. This can be interpreted as zoom, the smaller the crop the larger the zoom, specially for small objects."
        )
        crop_size_container = Container(widgets=[crop_size_label, self.crop_size_selector], layout="horizontal", labels=False)
        
        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

        threshold_label = Label(value="Segmentation Threshold:", name="subsection_label")
        self._threshold_slider = create_widget(
            annotation=float,
            widget_type="FloatSlider",
            value=0.5,
        )
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        self._threshold_slider.changed.connect(self._threshold_im)
        threshold_container = Container(widgets=[threshold_label, self._threshold_slider], labels=False)

        self._reset_btn = PushButton(
            text="Reset Default Settings",
            tooltip="Reset references and embeddings.",
        )
        self._reset_btn.changed.connect(self.reset_ref_and_emb)

        return Container(widgets=[
            image_section_label,
            image_layer_container,
            crop_size_container,
            threshold_container,
            self._reset_btn,
        ], labels=False)

    def _create_batch_processing_section(self):
        """Create the batch processing section of the GUI."""
        batch_processing_label = Label(value="Batch Processing", name="section_label")
        batch_processing_label.native.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._process_all_btn = PushButton(
            text="Process All Images",
            tooltip="Process all images with the given reference.",
        )
        self._process_all_btn.changed.connect(self.process_all)
        self._process_all_btn.native.setStyleSheet("background-color: darkgrey; color: black;")
        self._process_all_btn.enabled = False
        self.num_image_layers = 0

        return Container(widgets=[
            batch_processing_label,
            self._process_all_btn
        ], labels=False)

    def _new_image_selected(self):
        self.pipeline_engine.delete_precomputed_embeddings()
        self.auto_precompute()
        self._get_dist_map()

    def process_all(self):
        if self.pipeline_engine is None:
            return
        
        # Store current state
        self.original_layers = list(self._viewer.layers)
        
        # Update UI from main thread
        self._prepare_ui_for_processing_all()
        
        # Create worker for processing
        worker = create_worker(self._process_all_images)
        worker.finished.connect(self._on_processing_all_finished)
        worker.yielded.connect(self._on_image_processed)
        worker.start()

    def _prepare_ui_for_processing_all(self):
        """Prepare UI elements before processing starts"""
        self._image_layer_combo.reset_choices()
        self._viewer.layers.events.inserted.disconnect(self._on_layer_inserted)
        self._viewer.layers.events.removed.disconnect(self._on_layer_removed)
        self._process_all_btn.native.setStyleSheet("background-color: yellow; color: black;")
        self._process_all_btn.text = "Processing..."
        self._process_all_btn.enabled = False

    def _process_all_images(self):
        """Worker function to process all images"""
        crop_x, crop_y = self.crop_sizes[self.crop_size_selector.value]
        
        for layer in self.original_layers:
            if isinstance(layer, Image):
                # Process each image layer sequentially
                self.pipeline_engine.delete_precomputed_embeddings()
                image = self._get_nhwc_image(layer.data)
                image = self._touint8(image)
                self.pipeline_engine.pre_compute_embeddings(
                    image, overlap=(0, 0), padding=(0, 0), 
                    crop_shape=(crop_x, crop_y, image.shape[-1]), 
                    verbose=True, batch_size=1
                )
                self._get_dist_map(apply_threshold=False)
                yield layer.name

    def _on_image_processed(self, layer_name):
        self.threshold_im(layer_name)

    def _on_processing_all_finished(self):
        """Callback when all processing is complete"""
        # Re-enable UI elements
        self._process_all_btn.native.setStyleSheet("background-color: grey; color: black;")
        self._process_all_btn.text = "Process All Images"
        self._process_all_btn.enabled = True
        self._viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)

    def _check_existing_image_and_preprocess(self):
        """Check for existing image layers and preprocess if found."""
        image_found = False
        points_found = False
        for layer in self._viewer.layers:
            if not image_found and isinstance(layer, Image):
                self._image_layer_combo.reset_choices()
                self._image_layer_combo.value = layer
                self.auto_precompute()
                image_found = True
                # Process the first found image layer

            if not points_found and isinstance(layer, Points):
                self._points_layer = layer
                self._points_layer.events.data.connect(self._update_reference_and_process)
                points_found = True
                # Process the first found points layer

            if image_found and points_found:
                self._update_reference_and_process()
                break

        if image_found and not points_found:
            self._add_points_layer()

        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self.num_image_layers += 1
                if self.num_image_layers > 1:
                    self._process_all_btn.enabled = True
                    self._process_all_btn.native.setStyleSheet("color: white;")

    @thread_worker(start_thread=True)
    def precompute_threaded(self):
        self.auto_precompute()
    
    def auto_precompute(self):
        """Automatically precompute embeddings if the engine is available."""
        if self.pipeline_engine is not None:
            image_layer = self._image_layer_combo.value # (n),h,w,(c)
            if image_layer is not None:
                image = self._get_nhwc_image(image_layer.data)
                #image = ((image / np.iinfo(image.dtype).max) * 255).astype(np.uint8)
                assert image.shape[-1] in [1,3], f"{image.shape[-1]} channels are not allowed, only 1 or 3"
                image = self._touint8(image)
                if not self.pipeline_engine.emb_precomputed:
                    self.loaded_img_layer = self._image_layer_combo.value
                    crop_x, crop_y = self.crop_sizes[self.crop_size_selector.value]
                    self.pipeline_engine.pre_compute_embeddings(
                        image, overlap = (0, 0), padding=(0, 0), crop_shape=(crop_x, crop_y, image.shape[-1]), verbose=True, batch_size=1
                    )
                    
    def _touint8(self, image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 format with proper normalization.
        
        Parameters
        ----------
        image : np.ndarray
            Input image array
            
        Returns
        -------
        np.ndarray
            Converted uint8 image
        """
        if image.dtype != np.uint8:
            if 0 <= image.min() and image.max() <= 255:
                pass
            else:
                if not (0 <= image.min() <= 1 and 0 <= image.max() <= 1):
                    image = image - image.min()
                    image = image / image.max()
                image = image * 255
        return image.astype(np.uint8)

    def _get_nhwc_image(self, image):
        """Convert image to NHWC format."""
        image = np.squeeze(image)
        if len(image.shape) == 2:
            image = image[np.newaxis, ..., np.newaxis]
        elif len(image.shape) == 3:
            if image.shape[-1] in [3, 4]:
                # consider (h,w,c) rgb or rgba
                image = image[np.newaxis, ...]
            else:
                # consider 3D (n,h,w)
                image = image[..., np.newaxis]
        return image

    def reset_ref_and_emb(self):
        """Reset references and embeddings."""
        if self.pipeline_engine is not None:
            self._threshold_slider.value = 0.5
            self.pipeline_engine.delete_references()
            self.pipeline_engine.delete_precomputed_embeddings()
            self.precompute_threaded()

    def _get_dist_map(self, apply_threshold=True):
        """Generate and display the thresholded distance map."""
        if not self._references_coord:
            self._viewer.status = "No reference points selected"
            return
            
        if self.pipeline_engine is None:
            self._viewer.status = "Model not loaded"
            return

        try:
            distances = self.pipeline_engine.get_ds_distances_sameRef(
                use_euclidean_distance=self.use_euclidean_dist, 
                verbose=False
            )
            self.predictions = self.pipeline_engine.distance_post_processing(
                distances, 
                self.post_processings['gaussian'], 
                upsampling_mode=self.upsample,
                euclidean_distances=self.use_euclidean_dist
            )
            if apply_threshold:
                self._threshold_im()
        except Exception as e:
            self._viewer.status = f"Error processing image: {str(e)}"

    def _threshold_im(self):
        # simple callback, otherwise numeric value is given as parameter
        self.threshold_im()

    def threshold_im(self, file_name=None):
        """Apply the threshold to the prediction map and update the viewer."""
        if self.predictions is not None:
            thresholded = self.predictions > self._threshold_slider.value
            thresholded = np.squeeze(thresholded * 255).astype(np.uint8)
            name = self._image_layer_combo.value.name if file_name is None else file_name
            name += "_mask"

            if name in self._viewer.layers:
                self._viewer.layers[name].data = thresholded
            else:
                self._viewer.add_labels(thresholded, name=name)

    def _update_reference_and_process(self):
        """Update the reference coordinates and process the image."""
        points_layer = self._points_layer
        if points_layer is None:
            return

        image_layer = self._image_layer_combo.value
        if image_layer is not None:
            image = self._get_nhwc_image(image_layer.data)
            points = np.array(points_layer.data, dtype=int)
            n, h, w, c = image.shape
            # Compute mean color of the selected points
            self._references_coord = []
            for point in points:
                z, y, x = point if n > 1 else (0, *point)  # Handle 3D and 2D
                if 0 <= x < w and 0 <= y < h and 0 <= z < n:
                    self._references_coord.append((z, x, y))

            if self.pipeline_engine is not None and len(self._references_coord) > 0:
                self.precompute_threaded()
                self.pipeline_engine.set_reference_vector(
                    list_coords=self._references_coord, 
                    use_mean=self.use_references_coord
                )
                self._get_dist_map()
        
    def _img_processing_f(self, x):
        # input  tensor: [b,h,w,c] uint8 (0-255)
        # output tensor: [b,c,h,w] float32 (0-1)

        if x.shape[-1] == 1:
            x = x.repeat(1, 1, 1, 3)
        elif x.shape[-1] == 4:
            x = x[..., :3]
        x = x.permute(0, 3, 1, 2)

        x = x / 255
        return self.trfm(x)

    @thread_worker(start_thread=True)
    def _load_model(self):
        """Load the selected model based on the user's choice."""
        try:
            model_size = self.model_size_selector.value
            model_letter = model_size[0]
            
            if self.feat_dim != self.model_dims[model_size]:
                if self.model is not None:
                    self.model = None
                    cuda.empty_cache()

                self._load_model_btn.native.setStyleSheet("background-color: yellow; color: black;")
                self._load_model_btn.text = "Loading model..."

                self.model = hub.load('facebookresearch/dinov2', f'dinov2_vit{model_letter}14_reg')
                self.model.to(self.device)
                self.model.eval()

                self.feat_dim = self.model_dims[model_size]

                self._load_model_btn.native.setStyleSheet("background-color: lightgreen; color: black;")
                self._load_model_btn.text = f'Load New Model\n(Current Model: {model_size})'

                if self.pipeline_engine is not None:
                    self.pipeline_engine = None

                self.pipeline_engine = DinoSim_pipeline(
                    self.model,
                    self.model.patch_size,
                    self.device,
                    self._img_processing_f,
                    self.feat_dim,
                    dino_image_size=self.resize_size
                )

                self._check_existing_image_and_preprocess()
        except Exception as e:
            self._viewer.status = f"Error loading model: {str(e)}"
    
    def _add_points_layer(self):
        if self._points_layer == None:
            points_layer = self._viewer.add_points(data=None, size=10, name='Points Layer')
            points_layer.mode = 'add'
            self._viewer.layers.selection.active = self._viewer.layers['Points Layer']
            
    def _on_layer_inserted(self, event):
        try:
            layer = event.value

            if isinstance(layer, Image):
                self.num_image_layers += 1
                if self.num_image_layers > 1:
                    self._process_all_btn.enabled = True
                    self._process_all_btn.native.setStyleSheet("color: white;")
                else:
                    # Reset choices before setting new value
                    self._image_layer_combo.reset_choices()
                    self._image_layer_combo.value = layer
                    # Start precomputation and add points layer after
                    worker = self.precompute_threaded()
                    worker.finished.connect(lambda: self._add_points_layer())
                    worker.start()

            elif isinstance(layer, Points):
                if self._points_layer is not None:
                    self._points_layer.events.data.disconnect(self._update_reference_and_process)
                self._points_layer = layer
                self._points_layer.events.data.connect(self._update_reference_and_process)
        except Exception as e:
            print(e)
            self._viewer.status = f"Error: {str(e)}"

    def _on_layer_removed(self, event):
        layer = event.value

        if isinstance(layer, Image):
            if self.pipeline_engine != None and self.loaded_img_layer == layer:
                self.pipeline_engine.delete_precomputed_embeddings()
                self.loaded_img_layer = ""
            self.num_image_layers -= 1
            if self.num_image_layers < 2:
                self._process_all_btn.enabled = False
                self._process_all_btn.native.setStyleSheet("background-color: darkgrey; color: black;")
            self._image_layer_combo.reset_choices()

        elif layer is self._points_layer:
            self._points_layer.events.data.disconnect(self._update_reference_and_process)
            self._points_layer = None
            if self.pipeline_engine != None:
                self.pipeline_engine.delete_references()

    def closeEvent(self, event):
        """Clean up resources when widget is closed."""
        if self.pipeline_engine is not None:
            self.pipeline_engine.delete_precomputed_embeddings()
            self.pipeline_engine.delete_references()
            self.pipeline_engine = None

        if self.model is not None:
            self.model = None
            cuda.empty_cache()
        
        if self._points_layer is not None:
            self._points_layer.events.data.disconnect(self._update_reference_and_process)
            self._points_layer = None
            
        super().closeEvent(event)
