import os
import torch
import numpy as np

from affine import Affine
from rasterio.crs import CRS

import warnings
import rasterio
import rioxarray
import xarray as xr
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def assign_georeference(output, properties):
    transform = Affine.from_gdal(*(float(x) for x in properties['transform']))
    crs = CRS.from_string(properties['crs'][0])
    width, height = properties['width'][0], properties['height'][0]

    x_coords = np.linspace(float(transform.c),
                       float(transform.c + transform.a * (width - 1)),
                       int(width)).flatten()
    y_coords = np.linspace(float(transform.f),
                        float(transform.f + transform.e * (height - 1)),
                        int(height)).flatten()

    output_rioxarray = xr.DataArray(
        output,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, output.shape[0] + 1),
            "x": (["x"], x_coords),
            "y": (["y"], y_coords),
        }
    )

    # Write the transform and CRS
    output_rioxarray.rio.write_transform(transform, inplace=True)
    output_rioxarray.rio.write_crs(crs, inplace=True)
    
    return output_rioxarray


def imwrite_rasterio(img, file_path, auto_mkdir=True, dtype='uint16'):
    """Write image to file using rasterio, without geospatial metadata.

    Args:
        img (ndarray): Image array to be written (with values in the range 0-563).
        file_path (str): Image file path.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.
        dtype (str): Data type for saving the image (default: 'uint16').

    Raises:
        IOError: If writing the image fails.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    # Ensure the image is 3D (for multi-band images)
    if len(img.shape) == 2:  # Convert single-channel image to 3D
        img = img[np.newaxis, ...]

    # Define a minimal profile without CRS or transform
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': img.shape[0],  # Number of bands
        'width': img.shape[2],
        'height': img.shape[1],
    }

    try:
        with rasterio.open(file_path, 'w', **profile) as dst:
            for i in range(img.shape[0]):  # Write each band
                dst.write(img[i, :, :], i + 1)
    except Exception as e:
        raise IOError(f'Failed in writing images: {e}')


def tensor2img(tensor, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 563]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        if out_type == np.uint16:
            img_np = (img_np * 563.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


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


def imfrompath(content_path, float32=True, guide_data="biomass"):
    """Read an image from bytes.

    Args:
        content_path (str): Image path.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: True.
        guide_data (str): Whether we are reading biomass data, sentinel2 or landsat5. 
            Default: "biomass". Options: ["biomass", "sentinel2", "landsat5"]

    Returns:
        ndarray: Loaded image array.
    """
    allowed_types = {"biomass", "sentinel2", "landsat5", "none"}
    assert guide_data in allowed_types, f"Invalid data_type: {guide_data}. Must be one of {allowed_types}."
    
    # read raster
    with rasterio.open(content_path) as dataset:
        img = dataset.read()
    img = np.moveaxis(img, 0, -1) # CHW to HWC

    # define min and max values for the different types of data
    guide_ranges = {
        "biomass": (0., 563.),
        "sentinel2": (1., 10000.),
        "landsat5": (0., 1.),
        "none": (np.min(img), np.max(img))
    }
    min_value, max_value = guide_ranges.get(guide_data, (None, None))

    # (height, width) to (height, width, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    
    if float32:
        if guide_data != "none":
            if guide_data != "biomass":
                img = np.clip(img, a_min=min_value, a_max=max_value)
            img = (img - min_value) / (max_value - min_value) # normalize values
        img = img.astype(np.float32)
        
    return img