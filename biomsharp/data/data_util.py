from os import path as osp

from basicsr.utils import scandir

"""
This code is adapted from the BasicSR library (version 1.3.4.9), available at:
https://pypi.org/project/basicsr/1.3.4.9/

Original code was designed to handle two types of data (low resolution and high resolution).
This version has been modified to handle three types of data:
1) Input (input / lq)
2) Ground Truth (gt)
3) Guide (guide)
"""


def triple_paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding guide, lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder, guide_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt', 'guide'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder, guide_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key, guide_key]. But got {len(keys)}'
    input_folder, gt_folder, guide_folder = folders
    input_key, gt_key, guide_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb') and guide_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder, {gt_key} folder, and {guide_key} folder should all in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder};'
                         f'{guide_key}: {guide_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(guide_folder, 'meta_info.txt')) as fin:
        guide_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys) or set(input_lmdb_keys) != set(guide_lmdb_keys) or set(guide_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder, {gt_key}_folder and {guide_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key), (f'{guide_key}_path', lmdb_key)]))
        return paths


def triple_paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder, guide_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt', 'guide'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder, guide_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key, guide_key]. But got {len(keys)}'
    input_folder, gt_folder, guide_folder = folders
    input_key, gt_key, guide_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        guide_name = f'{filename_tmpl.format(basename)}{ext}'
        guide_path = osp.join(guide_folder, guide_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{guide_key}_path', guide_path)]))
    return paths


def triple_paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder, guide_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt', 'guide'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder, guide_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, f'The len of keys should be 3 with [input_key, gt_key, guide_key]. But got {len(keys)}'
    input_folder, gt_folder, guide_folder = folders
    input_key, gt_key, guide_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    guide_paths = list(scandir(guide_folder))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    assert len(input_paths) == len(guide_paths), (f'{input_key} and {guide_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(guide_paths)}.')
    assert len(guide_paths) == len(gt_paths), (f'{guide_key} and {gt_key} datasets have different number of images: '
                                               f'{len(guide_paths)}, {len(gt_paths)}.')
    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, f'{input_name} is not in {input_key}_paths.'
        guide_path = osp.join(guide_folder, input_name)
        assert input_name in guide_path, f'{input_name} is not in {guide_key}_paths.'
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{guide_key}_path', guide_path)]))
    return paths