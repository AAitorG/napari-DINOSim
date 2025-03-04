# DINOSim: Zero-Shot Object Detection and Segmentation

DINOSim is a powerful and easy-to-use plugin that helps you detect and segment objects in your images without requiring any training data. Using the advanced DINOv2 foundation model, it can identify and segment objects simply by clicking on them!

## 🎯 What can I use it for?

- Detecting and segmenting similar objects across multiple biomedical images
- Finding structures of interest in microscopy data
- Analyzing complex biological datasets where traditional segmentation methods might struggle
- Working with limited or no training data

## 🚀 Quick Start

1. Load your image into napari
2. Click on the object you want to segment
3. That's it! DINOSim will automatically generate a segmentation mask

Multiple clicks on the same type of object are supported to improve results.

## ⚙️ Adjustable Parameters

While DINOSim works great out of the box, you can fine-tune these parameters if needed:

- **Model Size**: Choose between different DINOv2 model sizes. Larger models are more accurate but require more computational resources.
- **Threshold**: Control how strict the segmentation should be. Higher values make it more selective.
- **Patch Size**: Adjust the detail level of the segmentation.

## 💡 Tips

- GPU acceleration is automatically used when available for faster processing
- You can process multiple images at once using the "Process All Images" button
- Use the "Reset" button to try different threshold values
- The plugin works best with clear, in-focus images

## 🎓 Example Use Cases

- Cell detection in microscopy images
- Tissue segmentation in histology
- Organelle detection in electron microscopy
- Pattern recognition in fluorescence microscopy

DINOSim makes complex segmentation tasks as simple as point and click!
