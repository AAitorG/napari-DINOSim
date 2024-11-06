from tifffile import imread
import cv2
from os.path import splitext

import numpy as np
from torchvision import transforms
from skimage.measure import label, regionprops
import torchvision.transforms.functional as T

def get_transforms(
        resize_size = 224,
        interpolation = transforms.InterpolationMode.BICUBIC,
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225),
        ):
        
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation, antialias=True),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

def gaussian_kernel(size=3, sigma=1):

    upper = size - 1
    lower = - int(size / 2)
    
    y, x = np.mgrid[lower:upper, lower:upper]
    
    kernel = (1 / (2 * np.pi * sigma**2 ) ) * np.exp( -(x**2 + y**2) / (2 * sigma**2) )
    kernel = kernel / kernel.sum()
    
    return kernel

def get_bounding_boxes(mask):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    bounding_boxes = [prop.bbox for prop in props]
    return bounding_boxes

def load_image(image_path):
    filename, file_extension = splitext(image_path)
    if file_extension[1:] in ['tif', 'tiff']:
        image = imread(image_path)
    else:
        # returns a list with the normalized image or patches 
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    return np.squeeze(image)

def resizeLongestSide(np_image, new_longest_size):
    h,w,*_ = np_image.shape
    scale = new_longest_size / max(h,w)
    hNew, wNew = h*scale, w*scale
    new_shape = (int(hNew+0.5), int(wNew+0.5))
    return np.array(T.resize(T.to_pil_image(np_image), new_shape))

def mirror_border(image, sizeH, sizeW):
    h_res = sizeH - image.shape[0]
    w_res = sizeW - image.shape[1]

    top = bot = h_res // 2
    left = right = w_res // 2
    top += 1 if h_res % 2 != 0 else 0
    left += 1 if w_res % 2 != 0 else 0

    res_image = np.pad(image, ( (top, bot), (left, right), (0,0)), 'symmetric')
    return res_image

def remove_padding(np_img, out_shape):
    '''
    Given an image and the shape of the original image, remove the padding from the image
    
    Args:
      np_img: the image to remove padding from
      out_shape (int,int): the desired shape of the output image (height, width)
    
    Returns:
      The image with the padding removed.

    Note:
        If returned image contain any 0 in the shape may be due to the given shape is greater than actual image shape
    '''

    _, height, width, _ = out_shape # original dimensions
    _, pad_height, pad_width, _ = np_img.shape # dimensions with padding
    
    rm_left = int( (pad_width - width)/2 )
    rm_top = int( (pad_height - height)/2 )

    rm_right = pad_width - width - rm_left if rm_left != 0 else -pad_width
    rm_bot = pad_height - height - rm_top if rm_top != 0 else -pad_height

    return np_img[:, rm_top:-rm_bot, rm_left:-rm_right, :]