# DINOSim

[![License MIT](https://img.shields.io/pypi/l/napari-dinoSim.svg?color=green)](https://github.com/AAitorG/napari-dinoSim/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-dinoSim.svg?color=green)](https://pypi.org/project/napari-dinoSim)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-dinoSim.svg?color=green)](https://python.org)
[![tests](https://github.com/AAitorG/napari-dinoSim/workflows/tests/badge.svg)](https://github.com/AAitorG/napari-dinoSim/actions)
[![codecov](https://codecov.io/gh/AAitorG/napari-dinoSim/branch/main/graph/badge.svg)](https://codecov.io/gh/AAitorG/napari-dinoSim)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-dinoSim)](https://napari-hub.org/plugins/napari-dinoSim)

## Description

This repository provides **DINOSim**, a method that leverages the DINOv2 foundation model for zero-shot object detection and segmentation in bioimage analysis. DINOSim uses pretrained DINOv2 embeddings to compare patch similarities, allowing it to detect and segment unseen objects in complex datasets with minimal annotations.

The **DINOSim Napari plugin** offers a user-friendly interface that simplifies bioimage analysis workflows, making it an adaptable solution for object detection across scientific research fields with limited labeled data.

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

One [notebook](./src/dinoSim_example.ipynb) example is provided in the repository to show how to use DINOSim directly through python, without napari.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

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

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-dinoSim" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[conda]: https://docs.conda.io/en/latest/miniconda.html
