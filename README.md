# DINOSim

[![License MIT](https://img.shields.io/pypi/l/napari-dinosim.svg?color=blue)](https://github.com/AAitorG/napari-DINOSim/raw/main/LICENSE)
[![biorxiv](https://img.shields.io/badge/bioRxiv-Paper-bd2635.svg)](https://doi.org/10.1101/2025.03.09.642092)
[![PyPI](https://img.shields.io/pypi/v/napari-DINOSim.svg?color=green)](https://pypi.org/project/napari-DINOSim)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-DINOSim.svg?color=green)](https://python.org)
[![tests](https://github.com/AAitorG/napari-DINOSim/workflows/tests/badge.svg)](https://github.com/AAitorG/napari-DINOSim/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-dinosim)](https://napari-hub.org/plugins/napari-dinosim)

![DINOSim-simple](docs/DINOSim-simplest.png)

A [napari] plugin for zero-shot image segmentation using DINOv2 vision transformers.

----------------------------------

## Overview

`napari-DINOSim` enables zero-shot image segmentation by selecting reference points on an image. The plugin leverages DINOv2's powerful feature extraction capabilities to compute similarity maps and generate segmentation masks.

For detailed information about the widget's functionality, UI elements, and usage instructions, please refer to the [Plugin Documentation](./docs/plugin_documentation.md). A simple [example notebook](./src/DINOSim_example.ipynb) demonstrating how to use DINOSim programmatically is also available.

## Installation

You can install `napari-DINOSim` via [pip]:

```sh
pip install napari-dinosim
```

or from source using [conda]:

```bash
# Clone the repository
git clone https://github.com/AAitorG/napari-DINOSim.git
cd napari-DINOSim

# Create and activate the conda environment
conda env create -f environment.yml
conda activate napari-dinosim
```

## Usage

To launch napari, run the following command in your terminal:

```sh
napari
```

Within the napari interface, locate and click the `DINOSim segmentation` plugin in the Plugins section of the top bar. You can then:
1. Drag and drop your image into the napari viewer
2. Select points on the objects you want to segment
3. The plugin will automatically generate segmentation masks based on your selections

For more detailed instructions and examples, please refer to our [Plugin Documentation](./docs/plugin_documentation.md).

## License

Distributed under the terms of the [MIT] license,
"napari-DINOSim" is free and open source software.

## Citation

Please note that DINOSim is based on a [publication](https://doi.org/10.1101/2025.03.09.642092). If you use DINOSim in your research, please be so kind to cite our work:

```bibtex
@article {Gonzalez-Marfil2025dinosim,
    title = {DINOSim: Zero-Shot Object Detection and Semantic Segmentation on Electron Microscopy Images},
    author = {Gonz{\'a}lez-Marfil, Aitor and G{\'o}mez-de-Mariscal, Estibaliz and Arganda-Carreras, Ignacio},
    journal = {bioRxiv},
    publisher = {Cold Spring Harbor Laboratory},
    url = {https://www.biorxiv.org/content/early/2025/03/13/2025.03.09.642092},
    doi = {10.1101/2025.03.09.642092},
    year = {2025}
}
```

## Contributing

Contributions are very welcome! Tests can be run with [tox]. Please ensure the test coverage at least stays the same before submitting a pull request.

## Issues

If you encounter any problems, please [file an issue](https://github.com/AAitorG/napari-DINOSim/issues) along with a detailed description.

[napari]: https://github.com/napari/napari
[MIT]: http://opensource.org/licenses/MIT
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[conda]: https://docs.conda.io/en/latest/miniconda.html
