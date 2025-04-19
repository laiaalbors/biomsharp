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
class FranceBiomassDataset(data.Dataset):
    def __init__(self, opt):
        super(FranceBiomassDataset, self).__init__()
        self.opt = opt
        self.gt_file, self.lq_file, self.gd_file = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_gd']
        self.satellite = opt['guide_data']

        # Validate the 'satellite' argument
        valid_satellites = {'landsat5', 'sentinel2'}
        if self.satellite not in valid_satellites:
            raise ValueError(f"Invalid satellite: '{self.satellite}'. Must be one of {valid_satellites}.")


    def __len__(self):
        return 1


    def __getitem__(self, idx):
        # Open the AGB-30m raster
        agb_30m = rioxarray.open_rasterio(self.gt_file)
        
        # Open the AGB-100m raster
        agb_100m = rioxarray.open_rasterio(self.lq_file)
        agb_100m_w, agb_100m_h = agb_100m.shape[1:]
        # agb_res_x, agb_res_y = abs(agb_100m.rio.resolution()[0]), abs(agb_100m.rio.resolution()[1])

        # BioSHARP: Create the Multispectral-25m raster
        multispectral_25m = rioxarray.open_rasterio(self.gd_file)
        
        # Resamplejar multispectral_25m per tenir la 4 vegades més petita que agb_100m
        multispectral_25m = multispectral_25m.rio.reproject(
            dst_crs=agb_100m.rio.crs,  # Manté el mateix sistema de coordenades
            # resolution=(agb_res_x / 4, agb_res_y / 4),
            shape=(agb_100m_w*4, agb_100m_h*4),
            resampling=1  # Resampling bilinear (1) o pots provar altres com cúbic (2) o nearest (0)
        )

        # New properties of multispectral_25m
        # profile = multispectral_25m.rio.profile
        transform = multispectral_25m.rio.transform()
        crs = multispectral_25m.rio.crs
        width = multispectral_25m.rio.width
        height = multispectral_25m.rio.height
        properties_multispectral_25m = {'transform': transform.to_gdal(), 'crs': str(crs), 'width': width, 'height': height}

        # Interpolate multispectral data if needed
        if self.satellite=='sentinel2':
            agb_25m = agb_30m.rio.reproject_match(multispectral_25m, resampling=Resampling.nearest)
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
        gd_min_value, gd_max_value = guide_ranges.get(self.satellite, (None, None))
        bio_min_value, bio_max_value = guide_ranges.get('biomass', (None, None))

        # Transformations and normalizations
        gt = np.nan_to_num(agb_30m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
        gt = img2tensor(gt, bgr2rgb=False, float32=True)
        gt = (gt - bio_min_value) / (bio_max_value - bio_min_value)
        
        lq = np.nan_to_num(agb_100m, copy=False, nan=-1.0).transpose(1, 2, 0)
        lq = img2tensor(lq, bgr2rgb=False, float32=True)
        lq = (lq - bio_min_value) / (bio_max_value - bio_min_value)
        
        # BioSHARP
        gd = np.nan_to_num(multispectral_25m, copy=False, nan=0).transpose(1, 2, 0)
        gd = img2tensor(gd, bgr2rgb=False, float32=True)
        gd = np.clip(gd, a_min=gd_min_value, a_max=gd_max_value)
        gd = (gd - gd_min_value) / (gd_max_value - gd_min_value)
        
        return {'idx': idx, 'lq': lq, 'gt': gt, 'gd': gd, 'lq_path': self.lq_file, 'gt_path': self.gt_file, 'gd_path': self.gd_file, 'properties': properties_multispectral_25m}


@DATASET_REGISTRY.register()
class FranceBiomassNaiveDataset(data.Dataset):
    def __init__(self, opt):
        super(FranceBiomassNaiveDataset, self).__init__()
        self.opt = opt
        self.gt_file, self.lq_file, self.gd_file = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_gd']
    

    def __len__(self):
        return 1


    def __getitem__(self, idx):
        # Open the AGB-30m raster
        agb_30m = rioxarray.open_rasterio(self.gt_file)
        
        # Open the AGB-100m raster
        agb_100m = rioxarray.open_rasterio(self.lq_file)
        agb_100m_w, agb_100m_h = agb_100m.shape[1:]
        # agb_res_x, agb_res_y = abs(agb_100m.rio.resolution()[0]), abs(agb_100m.rio.resolution()[1])

        # # Baseline: Reproject to match the GT
        # pred_agb_30m = agb_100m.rio.reproject_match(agb_30m, resampling=Resampling.cubic).values

        # Define min and max values for the different types of data
        guide_ranges = {
            "biomass": (0., 563.),
            "sentinel2": (1., 10000.),
            "landsat5": (0., 1.)
        }
        bio_min_value, bio_max_value = guide_ranges.get('biomass', (None, None))

        # Transformations and normalizations
        gt = np.nan_to_num(agb_30m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
        gt = img2tensor(gt, bgr2rgb=False, float32=True)
        gt = (gt - bio_min_value) / (bio_max_value - bio_min_value)
        
        lq = np.nan_to_num(agb_100m, copy=False, nan=-1.0).transpose(1, 2, 0)
        lq = img2tensor(lq, bgr2rgb=False, float32=True)
        lq = (lq - bio_min_value) / (bio_max_value - bio_min_value)

        # # Baseline
        # upscaled_lq = np.nan_to_num(pred_agb_30m, copy=False, nan=-1.0).transpose(1, 2, 0)
        # upscaled_lq = img2tensor(upscaled_lq, bgr2rgb=False, float32=True)
        # upscaled_lq = (upscaled_lq - bio_min_value) / (bio_max_value - bio_min_value)
        upscaled_lq = np.repeat(np.repeat(lq, 4, axis=1), 4, axis=2)

        return {'idx': idx, 'lq': lq, 'gt': gt, 'lq_path': self.lq_file, 'gt_path': self.gt_file, 'upscaled_lq': upscaled_lq, 'properties': properties_multispectral_25m}