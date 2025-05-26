# napari-dinoSim Plugin Documentation

This document provides a detailed overview of the `napari-dinosim` widget, its components, and how to use them.

## Overview

The `napari-dinosim` widget provides a user interface within napari for performing zero-shot image segmentation using DINO vision transformer models. It allows users to:

- Load different DINOv2 models (small, base, large, giant).
- Select an image layer from the napari viewer.
- Pick reference points on the image.
- Compute and visualize similarity maps based on the selected reference points.
- Adjust processing parameters like scale factor and segmentation threshold.
- Save and load precomputed image embeddings.
- Save and load reference points.
- Optionally, use SAM2 for post-processing and instance segmentation if available.

## Main UI Sections

The widget is organized into several collapsible sections:

1.  **Model Selection**: For choosing and loading DINO models.
2.  **Settings**: For controlling image processing parameters, managing embeddings, and reference points.
3.  **SAM2 Post-Processing**: For utilizing SAM2 for mask refinement and instance generation (only available if SAM2 is installed).

---

### 1. Model Selection

This section allows you to manage the DINOv2 model used for feature extraction.

-   **Model Size**: A dropdown menu to select the DINOv2 model architecture.
    -   `small`: ViT-S/14
    -   `base`: ViT-B/14
    -   `large`: ViT-L/14
    -   `giant`: ViT-G/14

    Larger models may provide better results but require more computational resources (GPU memory and processing time). The `small` model is loaded by default.
-   **Load Model Button**: Click this button to download (if not already cached by PyTorch Hub) and load the selected model onto the available compute device (GPU if available, otherwise CPU). The button text indicates the current status (e.g., "Load Model", "Loading model...", "Load New Model (Current: small)").
-   **GPU Status**: A non-interactive checkbox indicating whether a CUDA or MPS compatible GPU is available for computation.

---

### 2. Settings

This section contains controls for image selection, processing parameters, reference point management, and embedding management.

#### Reference Information (Collapsible)

-   **Reference Image**: Displays the name of the image layer used for reference point selection.
-   **Reference Points**: Displays the name of the points layer used for reference.
-   **Save Reference**: Saves the currently selected reference points and their computed feature vectors to a `.pt` file. This allows you to reuse the same reference for other images or sessions.
-   **Load Reference**: Loads previously saved reference points and feature vectors from a `.pt` file. This will apply the loaded reference to the currently selected image.

#### Processing Parameters (Collapsible)

-   **Image to segment**:
    -   A dropdown menu to select the napari `Image` layer you want to segment.
    -   An **Embedding Status Indicator** shows the state of the image embeddings:
        -   **Green**: Embeddings are computed and ready. The indicator's shape changes into a circle.
        -   **Yellow**: Embeddings are currently being computed.
        -   **Red**: Embeddings are not available or need recomputation (e.g., after changing the image or scale factor).
-   **Scale Factor**: A `FloatSpinBox` to adjust the scaling factor for image processing.
    -   Range: 0.1 to 10.0. Default: 1.0.
    -   Higher values result in smaller effective crop sizes during embedding computation, effectively "zooming in" for finer details. Lower values use larger crops.
    -   Changing the scale factor requires recomputing embeddings.
-   **Segmentation Threshold**: A `FloatSlider` to adjust the threshold for generating the binary segmentation mask from the similarity map.
    -   Range: 0.0 to 1.0. Default: 0.5.
    -   Lower values indicate higher similarity. Thus, a lower threshold makes the segmentation more restrictive (selects only highly similar regions).

#### Embedding Controls (Collapsible)

-   **Auto Precompute**: A checkbox (default: True) to automatically precompute image embeddings when a new image is selected or the scale factor is changed.
-   **Precompute Now**: A button (enabled when "Auto Precompute" is off) to manually trigger the precomputation of image embeddings for the currently selected image and scale factor. The button text changes to "Precomputing..." during the process.
-   **Save Embeddings**: Saves the precomputed feature embeddings for the entire current image to a `.pt` file. This is useful for large images or if you want to experiment with different reference points without recomputing embeddings each time. The filename will typically include the image name, model size, and scale factor.
-   **Load Embeddings**: Loads precomputed image embeddings from a `.pt` file. This also attempts to restore the scale factor if it's encoded in the filename.

-   **Reset Default Settings**: A button to reset all processing parameters (scale factor, threshold) to their default values, clear existing reference points, and delete precomputed embeddings.

---

### 3. SAM2 Post-Processing (Conditional)

WARNING: This functionality is in an experimental stage.

This section is only available if the SAM2 library is successfully imported. It provides tools to refine DINO-Sim segmentations using SAM2 or generate instance masks.

-   **Enable SAM2**:
    -   A checkbox to enable/disable SAM2 post-processing.
    -   A **SAM2 Status Indicator** (a colored circle) shows the state of SAM2 masks:
        -   **Green**: SAM2 masks are loaded and ready for refinement/instance generation.
        -   **Yellow**: SAM2 processor is initializing or masks are being loaded.
        -   **Red**: SAM2 is not enabled, masks are not loaded, or an error occurred.
-   **Load SAM2 Masks**: A button to load precomputed SAM2 masks from a `.pt` file. These masks are typically generated by running the SAM2 model on the entire image beforehand.
-   **Generate Instances**: A button to generate instance segmentation. This uses the current DINO-Sim similarity map (thresholded) as prompts for the loaded SAM2 masks to delineate individual objects.

To install SAM2, execute the following commands:
```sh
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
```
For more details, please refer to the [SAM2 GitHub repository](https://github.com/facebookresearch/sam2).

#### Generate SAM2 Masks

WARNING: This process is very time and compute-intensive.

To generate and save SAM2 masks, you can use the [sam2_preprocessing](../src/sam2_preprocessing.py) script using following command:

```sh
python ./src/sam2_preprocessing.py --input_path "./directory"
```

The following arguments can be passed to the script:

- `--input_path`: (str, required)
  - Path to a single image or a directory of images.

- `--output_dir`: (str, default: "sam2_masks")
  - Directory to save the generated masks.

- `--model_type`: (str, default: "tiny")
  - SAM2 model type ('tiny', 'small', 'base', 'large').

- `--points_per_side`: (int, default: 16)
  - Number of points per side for the SAM2 model.

- `--cuda_device`: (str, default: "0")
  - CUDA device to use (e.g., '0', '1', or 'cpu').

## Workflow

1.  **Open Napari and Add Image**: Launch napari and drag-and-drop the image you want to segment into the viewer.
2.  **Activate DINOSim Widget**: Find and open the DINOSim widget from the napari plugins menu.
3.  **Select Model (Optional)**:
    -   The `small` DINOv2 model is loaded by default.
    -   If you wish to use a different model, select it from the "Model Size" dropdown in the "Model Selection" section and click "Load Model". Wait for the model to load (button text will change).
4.  **Select Image Layer**:
    -   In the "Settings" section, under "Processing Parameters", choose your image from the "Image to segment" dropdown.
    -   If "Auto Precompute" (in "Embedding Controls") is enabled, embeddings for the image will be computed automatically. You can monitor the progress with the embedding status indicator and the "Precompute Now" button's text/state if manual.
5.  **Adjust Scale Factor (Optional)**:
    -   If needed, change the "Scale Factor". This will trigger recomputation of embeddings if "Auto Precompute" is on. Otherwise, click "Precompute Now".
6.  **Set Reference Points**:
    -   A "Points Layer" will be automatically added to the napari viewer if one doesn't exist and no reference is loaded.
    -   Select the "Points Layer" in napari's layer list and use the point adding tool.
    -   Click on one or more representative regions (reference points) in your image.
    -   As you add/remove points, the similarity map and the thresholded segmentation mask will update automatically.
7.  **Adjust Segmentation Threshold**:
    -   Fine-tune the "Segmentation Threshold" slider in "Processing Parameters" to get the desired segmentation output. The mask layer (e.g., `imagename_mask`) will update in real-time.
8.  **Save/Load (Optional)**:
    -   **Embeddings**: Use "Save Embeddings" to store the current image's embeddings for later use. Use "Load Embeddings" to load them back, avoiding recomputation.
    -   **Reference**: Use "Save Reference" to store the current set of reference points and their features. Use "Load Reference" to apply a saved reference to the current or a new image (ensure embeddings for that image are computed).
9.  **SAM2 Post-Processing (Optional)**:
    -   If SAM2 is installed and you have precomputed SAM2 masks for your image:
        -   Check "Enable SAM2".
        -   Click "Load SAM2 Masks" and select your SAM2 mask file. The status indicator should turn green.
        -   The displayed segmentation will now be refined by the SAM2 masks.
        -   Click "Generate Instances" to create an instance segmentation layer (e.g., `imagename_instances`) based on the DINO-Sim result and SAM2 masks.

## Tips

-   For large images, precomputing and saving embeddings ("Save Embeddings") can save significant time if you plan to experiment with different reference points or thresholds later.
-   If SAM2 is not detected, ensure it's installed in your napari Python environment.
-   The console output in napari might show progress or error messages from the DINOSim widget.
-   The "Reset Default Settings" button is useful to quickly return to a clean state.
-   The plugin uses the most recently added points layer, enabling you to delete it and create new points layers as needed.
-   3D volumes are also supported.
-   Images can also be added after opening the plugin.
