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
from scipy.interpolate import griddata

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


def interpolate_patch(patch, nan_mask, padding_size=1):
    # Apply mirror padding to the patch
    padded_patch = np.pad(patch, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), mode='reflect')
    padded_nan_mask = np.pad(nan_mask, ((0, 0), (padding_size, padding_size), (padding_size, padding_size)), mode='reflect')
    
    a, b = np.indices(padded_patch.shape[1:])
    
    for band in range(padded_patch.shape[0]):
        valid_pixels = ~padded_nan_mask[band, :, :]
        if not np.any(valid_pixels):
            continue
        valid_coords = np.column_stack((a[valid_pixels], b[valid_pixels]))
        values = padded_patch[band, :, :][valid_pixels]
        
        padded_patch[band, :, :] = griddata(
            valid_coords,
            values,
            (a, b),
            method='linear',
            fill_value=0
        )
    return padded_patch[:, padding_size:-padding_size, padding_size:-padding_size]


def process_in_patches(img, nan_mask, patch_size=512, padding_size=1):
    bands, H, W = img.shape
    output = np.empty_like(img)
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Determine patch boundaries with overlap for padding
            i_start = max(i - padding_size, 0)
            i_end = min(i + patch_size + padding_size, H)
            j_start = max(j - padding_size, 0)
            j_end = min(j + patch_size + padding_size, W)
            
            patch = img[:, i_start:i_end, j_start:j_end]
            patch_nan = nan_mask[:, i_start:i_end, j_start:j_end]
            
            # Interpolate only if there are NaNs
            if np.any(patch_nan):
                patch_filled = interpolate_patch(patch, patch_nan, padding_size=padding_size)
            else:
                patch_filled = patch
                
            # Compute indices for placing the patch back
            out_i_start = i
            out_i_end = min(i + patch_size, H)
            out_j_start = j
            out_j_end = min(j + patch_size, W)
            
            # Calculate the slice of the patch to extract (remove overlap)
            pi_start = i - i_start
            pi_end = pi_start + (out_i_end - out_i_start)
            pj_start = j - j_start
            pj_end = pj_start + (out_j_end - out_j_start)
            
            output[:, out_i_start:out_i_end, out_j_start:out_j_end] = patch_filled[:, pi_start:pi_end, pj_start:pj_end]
    return output


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
        multispectral_25m = multispectral_img.rio.reproject_match(agb_25m, resampling=Resampling.nearest).values

        # Interpolate multispectral data if needed
        if self.satellite=='sentinel2':
            multispectral_nan_mask = np.any(np.isnan(multispectral_25m), axis=0)
            agb_values_at_nans = agb_25m.values[0][multispectral_nan_mask]
            if np.any(agb_values_at_nans > 0):
                nan_mask = np.isnan(multispectral_25m)
                multispectral_25m = process_in_patches(multispectral_25m, nan_mask, patch_size=1024, padding_size=1)
        
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
        
        gd = np.nan_to_num(multispectral_25m, copy=False, nan=0).transpose(1, 2, 0)
        gd = img2tensor(gd, bgr2rgb=False, float32=True)
        gd = np.clip(gd, a_min=gd_min_value, a_max=gd_max_value)
        gd = (gd - gd_min_value) / (gd_max_value - gd_min_value)
        
        return {'idx': idx, 'lq': lq, 'gt': gt, 'gd': gd, 'lq_path': row["filename_100m"], 'gt_path': row["filename_25m"], 'gd_path': multispectral_path}
        # Per alcular mae entre 100 i 25 tal qual:
        # return {'idx': idx, 'lq': lq, 'gt': gt, 'lq_path': row["filename_100m"], 'gt_path': row["filename_25m"], 'upscaled_lq': upscaled_lq}
