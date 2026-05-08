"""Utility functions for the napari-dinosim plugin."""

# Re-export commonly used functions from utils
# Pipeline
from .dinoSim_pipeline import DINOSim_pipeline

# GUI utilities
from .gui_utils import CollapsibleSection
from .utils import (
    ensure_valid_dtype,
    gaussian_kernel,
    get_img_processing_f,
    get_nhwc_image,
    load_image,
    mirror_border,
    remove_padding,
    resizeLongestSide,
    torch_convolve,
)

# SAM2 processor if available
try:
    from .sam2_utils import SAM2Processor

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

__all__ = [
    "gaussian_kernel",
    "get_img_processing_f",
    "torch_convolve",
    "ensure_valid_dtype",
    "get_nhwc_image",
    "load_image",
    "resizeLongestSide",
    "mirror_border",
    "remove_padding",
    "CollapsibleSection",
    "DINOSim_pipeline",
    "HAS_SAM2",
]

if HAS_SAM2:
    __all__.append("SAM2Processor")
