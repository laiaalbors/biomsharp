import rasterio
import numpy as np


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
    # define min and max values for the given bands --> hauria de distingir entre bio25 i bio100?
    max_values = 563. if biomass else np.array([10000]*img.shape[2])
    min_values = 0. if biomass else np.array([1]*img.shape[2])

    with rasterio.open(content_path) as dataset:
        img = dataset.read()
    img = np.moveaxis(img, 0, -1) # CHW to HWC

    # (height, width) to (height, width, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    if float32:
        img = (img - min_values) / (max_values - min_values) # normalize values
        img = img.astype(np.float32)
        
    return img