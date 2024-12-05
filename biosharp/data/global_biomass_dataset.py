import time
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils import data as data

import rioxarray
from rasterio.enums import Resampling

from biosharp.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


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
        agb_25m = agb_25m.rio.reproject(dst_width=8960, dst_height=8960, resampling=Resampling.bilinear, dst_crs=agb_25m.rio.crs)
        
        # Open the AGB-100m raster and reproject to match the reference
        agb_100m = rioxarray.open_rasterio(row["filename_100m"])
        minx, miny, maxx, maxy = agb_25m.rio.bounds()
        agb_100m = agb_100m.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        # agb_100m = agb_100m.rio.reproject(dst_width=2250, dst_height=2250, resampling=Resampling.bilinear, dst_crs=agb_100m.rio.crs)
        agb_100m = agb_100m.rio.reproject(dst_width=2240, dst_height=2240, resampling=Resampling.bilinear, dst_crs=agb_100m.rio.crs)

        # Create the Multispectral-25m raster and reproject to match the reference
        multispectral_path = row["filename_l5"] if self.satellite=='landsat5' else row["filename_s2"]
        multispectral_img = rioxarray.open_rasterio(multispectral_path)
        multispectral_25m = multispectral_img.rio.reproject_match(agb_25m)
        multispectral_25m = multispectral_25m.rio.reproject(dst_width=8960, dst_height=8960, resampling=Resampling.bilinear, dst_crs=multispectral_25m.rio.crs)
        
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
        gt = np.clip(gt, a_min=bio_min_value, a_max=bio_max_value)
        gt = (gt - bio_min_value) / (bio_max_value - bio_min_value)
        
        lq = np.nan_to_num(agb_100m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
        lq = img2tensor(lq, bgr2rgb=False, float32=True)
        lq = np.clip(lq, a_min=bio_min_value, a_max=bio_max_value)
        lq = (lq - bio_min_value) / (bio_max_value - bio_min_value)
        
        gd = np.nan_to_num(multispectral_25m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
        gd = img2tensor(gd, bgr2rgb=False, float32=True)
        gd = np.clip(gd, a_min=gd_min_value, a_max=gd_max_value)
        gd = (gd - gd_min_value) / (gd_max_value - gd_min_value)
        
        return {'idx': idx, 'lq': lq, 'gt': gt, 'gd': gd, 'lq_path': row["filename_100m"], 'gt_path': row["filename_25m"], 'gd_path': multispectral_path}
