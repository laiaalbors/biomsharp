import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

import rasterio
from scipy.interpolate import griddata, RegularGridInterpolator

from basicsr.utils import scandir


"""
This code is adapted from the BasicSR GitHub Repository, available at:
https://github.com/XPixelGroup/BasicSR/

Original code was designed to handle normal images.
This adaptation works with images of any number of channels,
and fills the NaNs values using linear interpolation.
"""


def main():
    """A multi-thread tool to crop large images to sub-images for faster IO.

    It is used for DIV2K dataset.

    Args:
        opt (dict): Configuration dict. It contains:
        n_thread (int): Thread number.
        compression_level (int):  CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and
            longer compression time. Use 0 for faster CPU decompression. Default: 3, same in cv2.
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.

    Usage:
        For each folder, run this script.
        Typically, there are four folders to be processed for DIV2K dataset.

            * DIV2K_train_HR
            * DIV2K_train_LR_bicubic/X2
            * DIV2K_train_LR_bicubic/X3
            * DIV2K_train_LR_bicubic/X4

        After process, each sub_folder should have the same number of subimages.

        Remember to modify opt configurations according to your settings.
    """

    opt = {}
    opt['n_thread'] = 10
    opt['compression_level'] = 3

    # HR images
    opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_HR'
    opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_HR_sub'
    opt['crop_size'] = 480
    opt['step'] = 240
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # LRx2 images
    opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2'
    opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    opt['crop_size'] = 240
    opt['step'] = 120
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # LRx4 images
    opt['input_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4'
    opt['save_folder'] = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    opt['crop_size'] = 120
    opt['step'] = 60
    opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
        input_folder (str): Path to the input folder.
        save_folder (str): Path to save folder.
        n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))
    print(f"Number of images to crop into subimages: {len(img_list)}")

    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    worker_with_opt = partial(worker, opt=opt)
    with Pool(opt['n_thread']) as pool:
        for _ in pool.starmap(worker_with_opt, zip(img_list)):
            pbar.update(1)
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def interpolate_with_padding(cropped_img, nan_mask, index, img_name, padding_size=1):
    if np.any(nan_mask):
        # Apply mirror padding to the image and the NaN mask
        padded_img = np.pad(cropped_img, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='reflect')
        padded_nan_mask = np.pad(nan_mask, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='reflect')

        # Create grid of indices
        a, b = np.indices((padded_img.shape[0], padded_img.shape[1]))

        # Interpolate only for NaN values
        for band in range(padded_img.shape[2]):
            valid_pixels = ~padded_nan_mask[:, :, band]  # Mask for valid pixels
            padded_img[:, :, band] = griddata(
                (a[valid_pixels], b[valid_pixels]),   # Coordinates of valid pixels
                padded_img[:, :, band][valid_pixels],        # Values of valid pixels
                (a, b),                               # Coordinates of all pixels
                method='linear',
                fill_value=0                     # Optionally keep border NaN if interpolation cannot fill them
            )
        
        # Remove the padding after interpolation to return to original size
        cropped_img_filled = padded_img[padding_size:-padding_size, padding_size:-padding_size, :]
        
        has_zero = (cropped_img_filled == 0).any()
        # has_nan = np.isnan(cropped_img_filled).any()
        if has_zero:
            print(f'Warning: NaN values still present in cropped_img {index} of {img_name}, put them to 0.', flush=True)
    
        return cropped_img_filled
    
    else:
        return cropped_img


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
        crop_size (int): Crop size.
        step (int): Step for overlapped sliding window.
        thresh_size (int): Threshold size. Patches whose size is lower than thresh_size will be dropped.
        save_folder (str): Path to save folder.
        compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    # remove the x2, x3, x4 and x8 in the filename for DIV2K
    img_name = img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')

    with rasterio.open(path) as dataset:
        img = dataset.read()
        profile = dataset.profile
    img = np.moveaxis(img, 0, -1) 

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1

            if not os.path.isfile(osp.join(opt['save_folder'], f'{img_name}_s{index:04d}{extension}')):
                cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
                cropped_img = np.ascontiguousarray(cropped_img)
                
                #------ interpolate missing values -------
                nan_mask = np.isnan(cropped_img)
                cropped_img = interpolate_with_padding(cropped_img, nan_mask, index, img_name)
                #-----------------------------------------
                
                profile.update(
                    height=cropped_img.shape[0],
                    width=cropped_img.shape[1],
                    count=cropped_img.shape[2] if len(cropped_img.shape) == 3 else 1,
                    compress='lzw'
                )

                # Save the cropped image using rasterio
                output_path = osp.join(opt['save_folder'], f'{img_name}_s{index:04d}{extension}')
                with rasterio.open(output_path, 'w', **profile) as dst:
                    if cropped_img.ndim == 2:
                        dst.write(cropped_img, 1)
                    else:
                        for band in range(cropped_img.shape[2]):
                            dst.write(cropped_img[:, :, band], band + 1)
    
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()