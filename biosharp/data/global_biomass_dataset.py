import time
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils import data as data

import rasterio
import rioxarray
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

from biosharp.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


def crop_to_match(file_25m, file_100m, mida=2250):
    """
    Crops the agb_100m raster to match the bounds of the agb_25m raster and returns the cropped result.

    Parameters:
        file_25m (str): Path to the 25m reference raster file.
        file_100m (str): Path to the 100m raster file to be cropped.
        mida (int): The fixed size of the cropped window (default: 480).

    Returns:
        np.ndarray: The cropped data from agb_100m.
        dict: Updated profile of the cropped raster.
    """
    with rasterio.open(file_25m) as src_25m:
        bounds_25m = src_25m.bounds

        with rasterio.open(file_100m) as src_100m:
            transform_100m = src_100m.transform

            # Calculate the initial window for cropping
            window = from_bounds(*bounds_25m, transform_100m)

            # Adjust window size to exactly mida x mida
            if window.width != mida or window.height != mida:
                # Calculate new offsets to ensure the window is centered around the original
                col_off = max(window.col_off - (mida - window.width) // 2, 0)
                row_off = max(window.row_off - (mida - window.height) // 2, 0)

                # Adjust offsets to avoid going out of bounds
                if col_off + mida > src_100m.width:
                    col_off = src_100m.width - mida
                if row_off + mida > src_100m.height:
                    row_off = src_100m.height - mida

                window = Window(col_off, row_off, mida, mida)

            profile = src_100m.profile
            profile.update({
                'height': mida,
                'width': mida,
                'transform': rasterio.windows.transform(window, transform_100m)
            })

            # Read the cropped area
            data = src_100m.read(window=window)

    return data


@DATASET_REGISTRY.register()
class GlobalBiomassDataset(data.Dataset):
    def __init__(self, opt):
        super(GlobalBiomassDataset, self).__init__()
        self.opt = opt
        csv_path = opt['csv_path']
        satellite = opt['guide_data']
        first_row = opt['first_row'] if 'first_row' in opt else 0
        last_row = opt['last_row'] if 'last_row' in opt else None
        self.guide_data = opt['guide_data'] # sentinel2 or landsat5

        # Validate the 'satellite' argument
        valid_satellites = {'landsat5', 'sentinel2'}
        if satellite not in valid_satellites:
            raise ValueError(f"Invalid satellite: '{satellite}'. Must be one of {valid_satellites}.")

        # data could be a list, DataFrame, or any other structured format
        self.df = pd.read_csv(csv_path)

        if last_row is None:
            last_row = len(self.df)
        
        self.df = self.df[first_row:last_row].reset_index(drop=True)

        # Drop rows with missing values
        self.df.dropna(inplace=True)

        self.satellite = satellite


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        # Retrieve data point at index `idx`
        row = self.df.iloc[idx]

        # Open the AGB-25m raster
        agb_25m = rioxarray.open_rasterio(row["filename_25m"])
        
        # Open the AGB-100m raster and reproject to match the reference
        agb_100m = crop_to_match(row["filename_25m"], row["filename_100m"])

        # Create the Multispectral-25m raster and reproject to match the reference
        multispectral_path = row["filename_l5"] if self.satellite=='landsat5' else row["filename_s2"]
        multispectral_img = rioxarray.open_rasterio(multispectral_path)
        multispectral_25m = multispectral_img.rio.reproject_match(agb_25m, resampling=Resampling.nearest)
        
        # Define min and max values for the different types of data
        guide_ranges = {
            "biomass": (0., 563.),
            "sentinel2": (1., 10000.),
            "landsat5": (0., 1.)
        }
        gd_min_value, gd_max_value = guide_ranges.get(self.guide_data, (None, None))
        bio_min_value, bio_max_value = guide_ranges.get('biomass', (None, None))

        # Transformations and normalizations
        gt = np.nan_to_num(agb_25m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
        gt = img2tensor(gt, bgr2rgb=False, float32=True)
        gt = (gt - bio_min_value) / (bio_max_value - bio_min_value)
        
        lq = np.nan_to_num(agb_100m, copy=False, nan=-1.0).transpose(1, 2, 0)
        lq = img2tensor(lq, bgr2rgb=False, float32=True)
        lq = (lq - bio_min_value) / (bio_max_value - bio_min_value)
        # Per calcular mae entre 100 i 25 tal qual:
        # upscaled_lq = np.repeat(np.repeat(lq, 4, axis=1), 4, axis=2)
        
        gd = np.nan_to_num(multispectral_25m.values, copy=False, nan=0).transpose(1, 2, 0)
        gd = img2tensor(gd, bgr2rgb=False, float32=True)
        gd = np.clip(gd, a_min=gd_min_value, a_max=gd_max_value)
        gd = (gd - gd_min_value) / (gd_max_value - gd_min_value)
        
        return {'idx': idx, 'lq': lq, 'gt': gt, 'gd': gd, 'lq_path': row["filename_100m"], 'gt_path': row["filename_25m"], 'gd_path': multispectral_path}
        # Per alcular mae entre 100 i 25 tal qual:
        # return {'idx': idx, 'lq': lq, 'gt': gt, 'lq_path': row["filename_100m"], 'gt_path': row["filename_25m"], 'upscaled_lq': upscaled_lq}
