<!-- This file is a placeholder for customizing description of your plugin 
on the napari hub if you wish. The readme file will be used by default if
you wish not to do any customization for the napari hub listing.

If you need some help writing a good description, check out our 
[guide](https://github.com/chanzuckerberg/napari-hub/wiki/Writing-the-Perfect-Description-for-your-Plugin)
-->

# DinoSim

## Description

This repository provides **DinoSim**, a method that leverages the DINOv2 foundation model for zero-shot object detection and segmentation in bioimage analysis. DinoSim uses pretrained DINOv2 embeddings to compare patch similarities, allowing it to detect and segment unseen objects in complex datasets with minimal annotations.

The **DinoSim Napari plugin** offers a user-friendly interface that simplifies bioimage analysis workflows, making it an adaptable solution for object detection across scientific research fields with limited labeled data.

**Note**: The current version of the plugin generates segmentation masks based on object similarity.

## Usage

<!--
Add usage instructions with screenshots/gifs
-->
To use the plugin, you only need to load an image and click in the object you want to segment, automatically a mask will be generated. Multiple prompts for the same object are allowed.

## Documentation
There are few parameters that you can modify to improve the segmentation results:

- **Model size**: This parameter controls the size of the DINOv2 model used. Larger models are more accurate but also require more memory and processing power. Once you have selected the model size, you need to click in **Load New Model** button to apply the changes. The button will indicate the model that is being used. By default, the pluggin will use the smallest model.
- **Threshold**: This parameter controls the minimum similarity score between the query patch and the reference patches. Adjusting this value allows you to control the sensitivity of the segmentation. Higher values make the segmentation more strict, while lower values make it more permissive.
- **Patch size**: This parameter controls the size of the patches used for segmentation. Adjusting this value allows you to control the granularity of the segmentation. Smaller values make the segmentation more detailed, while larger values make it more coarse.

Using GPU acceleration is recommended for faster processing. If its available, the plugin will use it automatically. To check if your GPU is being used, you can check the **GPU status** tab in the napari viewer.

If multiple images are loaded, you need to specify which one is the prompted, using the **Image layer**. Once you have the prompted image, you can find all object in all other images by using **Process All Images** button.

If you want to reset the threshold and process the reference image again, you can use the **Reset** button. This might be useful if you changed the **Image layer** and the model has not been applied automatically.

By default, the plugin will add a annotation layer, but if you remove it or add more than one, only the last one will be used. Until you do not prompt in the anotation layer with one loaded image, the pluggin will not give any output.

## Example

One [notebook](../src/dinoSim_example.ipynb) example is provided in the repository to show how to use DinoSim directly through python, without napari.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-dinoSim` either via [pip]:

    pip install napari-dinosim

or from source via [conda]:

```bash
# Clone the repository
git clone https://github.com/AAitorG/napari-dinoSim.git
cd napari-dinoSim

# Create and activate the conda environment
conda env create -f environment.yml
conda activate napari-dinosim
```
