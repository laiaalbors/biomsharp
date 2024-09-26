import torch
import rasterio
import numpy as np


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """
    
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img)
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def imfrompath(content_path, float32=True, biomass=True):
    """Read an image from bytes.

    Args:
        content_path (str): Image path.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: True.
        biomass (bool): Whether we are reading biomass data or not. Default: True. 

    Returns:
        ndarray: Loaded image array.
    """
    # read raster
    with rasterio.open(content_path) as dataset:
        img = dataset.read()
    img = np.moveaxis(img, 0, -1) # CHW to HWC

    # define min and max values for the given bands
    max_values = 563. if biomass else np.array([10000]*img.shape[2])
    min_values = 0. if biomass else np.array([1]*img.shape[2])

    # (height, width) to (height, width, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    if float32:
        img = (img - min_values) / (max_values - min_values) # normalize values
        img = img.astype(np.float32)
        
    return img