from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("napari-dinosim")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.1.4"

from ._widget import DINOSim_widget

__all__ = ("DINOSim_widget",)
