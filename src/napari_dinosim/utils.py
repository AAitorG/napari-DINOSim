from os.path import splitext

import numpy as np
import torchvision.transforms.v2.functional as T
from cv2 import imread
from tifffile import imread as tif_imread
from torch.nn import functional as F
from torchvision import transforms


def get_img_processing_f(
    resize_size=224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # input  tensor: [(b),h,w,c] uint8 (0-255)
    # output tensor: [(b),c,h,w] float32 (0-1)
    def _img_processing_f(x):
        if len(x.shape) == 4:
            if x.shape[-1] == 1:
                x = x.repeat(1, 1, 1, 3)
            x = x.permute(0, 3, 1, 2)
        else:
            if x.shape[-1] == 1:
                x = x.repeat(1, 1, 3)
            x = x.permute(2, 0, 1)
        x = x / 255
        x = T.resize(
            x, resize_size, interpolation=interpolation, antialias=True
        )
        x = T.normalize(x, mean=mean, std=std)
        return x

    return _img_processing_f


def gaussian_kernel(size=3, sigma=1):

    upper = size - 1
    lower = -int(size / 2)

    y, x = np.mgrid[lower:upper, lower:upper]

    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )
    kernel = kernel / kernel.sum()

    return kernel


def load_image(image_path):
    filename, file_extension = splitext(image_path)
    if file_extension[1:] in ["tif", "tiff"]:
        image = tif_imread(image_path)
    else:
        image = imread(image_path, -1)  # cv2.IMREAD_UNCHANGED
    return np.squeeze(image)


def resizeLongestSide(np_image, new_longest_size):
    h, w, *_ = np_image.shape
    scale = new_longest_size / max(h, w)
    hNew, wNew = h * scale, w * scale
    new_shape = (int(hNew + 0.5), int(wNew + 0.5))
    return np.array(T.resize(T.to_pil_image(np_image), new_shape))


def mirror_border(image, sizeH, sizeW):
    h_res = sizeH - image.shape[0]
    w_res = sizeW - image.shape[1]

    top = bot = h_res // 2
    left = right = w_res // 2
    top += 1 if h_res % 2 != 0 else 0
    left += 1 if w_res % 2 != 0 else 0

    res_image = np.pad(image, ((top, bot), (left, right), (0, 0)), "symmetric")
    return res_image


def remove_padding(torch_img, out_shape):
    """
    Given an image and the shape of the original image, remove the padding from the image

    Args:
      torch_img: the image to remove padding from (shape: b,c,h,w)
      out_shape (int,int): the desired shape of the output image (height, width)

    Returns:
      The image with the padding removed.

    Note:
        If returned image contain any 0 in the shape may be due to the given shape is greater than actual image shape
    """

    *_, height, width = out_shape  # original dimensions
    _, _, pad_height, pad_width = torch_img.shape  # dimensions with padding

    rm_left = int((pad_width - width) / 2)
    rm_top = int((pad_height - height) / 2)

    rm_right = pad_width - width - rm_left if rm_left != 0 else -pad_width
    rm_bot = pad_height - height - rm_top if rm_top != 0 else -pad_height

    return torch_img[:, :, rm_top:-rm_bot, rm_left:-rm_right]


def torch_convolve(input, weights, mode="reflect", cval=0.0, origin=0):
    """
    Multidimensional convolution using PyTorch.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be convolved.
    weights : torch.Tensor
        Convolution kernel, with the same number of dimensions as the input.
    mode : str, optional
        Padding mode. Options are 'reflect', 'constant', 'replicate', or 'circular'.
        Default is 'reflect'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'. Default is 0.0.
    origin : int, optional
        Controls the origin of the input signal. Positive values shift the filter
        to the right, and negative values shift the filter to the left. Default is 0.

    Returns
    -------
    result : torch.Tensor
        The result of convolution of `input` with `weights`.
    """
    # Ensure input is 4D (batch, channels, height, width)
    if input.dim() == 2:  # Single channel 2D image
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:  # Add batch dimension if missing
        input = input.unsqueeze(0)

    # Add channel dimension for weights if necessary
    if weights.dim() == 2:
        weights = weights.unsqueeze(0).unsqueeze(0)

    # Apply padding based on mode
    padding = (
        weights.shape[-1] // 2 - origin
    )  # Adjust padding for origin shift
    input_padded = F.pad(
        input, (padding, padding, padding, padding), mode=mode, value=cval
    )

    # Perform convolution
    result = F.conv2d(input_padded, weights)

    return result.squeeze()  # Remove extra dimensions for output
