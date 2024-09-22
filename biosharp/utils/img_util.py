import rasterio
import numpy as np


def imfrompath(content_path, float32=True, bands=["BIO"]):
    """Read an image from bytes.

    Args:
        content_path (str): Image path.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: True.
        bands (list[str]): list with the Sentinel-2 bands used as guidence. 
            Options: ["BIO", "02", "03", "04", "05", "06", "07", "08", "8A", "11", "12"]

    Returns:
        ndarray: Loaded image array.
    """
    # define mean values for the given bands --> hauria de posar les correctes i només de train? I hauria de distingir entre bio25 i bio100?
    dict_means_bands = {"BIO": 110, "02": 443.08962184334, "03": 689.82896346775, "04": 593.4743311779, "05": 1099.0769497062, "06": 2429.743125455, "07": 2832.3039480575, "08": 2925.0695331386, "8A": 3065.8751902624, "11": 1676.9821219789, "12": 947.05621559662}
    mean_values = np.array([dict_means_bands[band] for band in bands])

    # define max values for the given bands --> hauria de posar les correctes i només de train? I hauria de distingir entre bio25 i bio100?
    dict_maxs_bands = {"BIO": 563.,"02": 16243.5, "03": 16069.0, "04": 15657.5, "05": 15824, "06": 15452, "07": 15616, "08": 15459, "8A": 515515, "11": 14520, "12": 15197}
    max_values = np.array([dict_maxs_bands[band] for band in bands])

    # define min values for the given bands
    dict_mins_bands = {"BIO": 0,"02": 1, "03": 1, "04": 1, "05": 1, "06": 1, "07": 1, "08": 1, "8A": 1, "11": 1, "12": 1}
    min_values = np.array([dict_mins_bands[band] for band in bands])

    with rasterio.open(content_path) as dataset:
        img = dataset.read()
    img = np.moveaxis(img, 0, -1) 

    # (height, width) to (height, width, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    if float32:
        for channel in range(img.shape[2]):
            img[channel][img[channel] == -1] = mean_values[channel] # nan values to mean
        img = (img - min_values) / (max_values - min_values) 
        img = img.astype(np.float32)
        
    return img