from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, FileClient # TODO: vaig haver de canviar una línia de la funció img2tensor original, hauria de comprovar si encara peta!
from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY

from biosharp.data import augment, triplet_random_crop, triple_paired_paths_from_lmdb, triple_paired_paths_from_meta_info_file, triple_paired_paths_from_folder
from biosharp.utils import imfrompath, img2tensor


@DATASET_REGISTRY.register()
class TripletImageDataset(data.Dataset):
    """Triplet image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc), GD (guide) and GT image triplets.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_gd (str): Data root path for gd.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(TripletImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_gdt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.mean_gd = opt['mean_gd'] if 'mean_gd' in opt else None
        self.std_gd = opt['std_gd'] if 'std_gd' in opt else None
        # TODO: Afegir "sentinel_bands" al fitxer de configuració
        self.bands = eval(opt['sentinel_bands']) if 'sentinel_bands' in opt else ["04", "03", "02"] # RGB

        self.gt_folder, self.lq_folder, self.gd_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_gd']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_gdt['type'] == 'lmdb':
            self.io_backend_gdt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_gdt['client_keys'] = ['lq', 'gt', 'gd']
            self.paths = triple_paired_paths_from_lmdb([self.lq_folder, self.gt_folder, self.gd_folder], ['lq', 'gt', 'gd'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = triple_paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder, self.gd_folder], ['lq', 'gt', 'gd'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = triple_paired_paths_from_folder([self.lq_folder, self.gt_folder, self.gd_folder], ['lq', 'gt', 'gd'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_gdt.pop('type'), **self.io_backend_gdt)

        scale = self.opt['scale']

        # Load gd, gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        # GT
        gt_path = self.paths[index]['gt_path']
        img_gt = imfrompath(gt_path, float32=True)
        # LQ
        lq_path = self.paths[index]['lq_path']
        img_lq = imfrompath(lq_path, float32=True)
        # GD
        gd_path = self.paths[index]['gd_path']
        img_gd = imfrompath(gd_path, float32=True, biomass=False)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, img_gd = triplet_random_crop(img_gt, img_lq, img_gd, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_gd = augment([img_gt, img_lq, img_gd], self.opt['use_hflip'], self.opt['use_rot'])

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            img_gd = img_gd[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # HWC to CHW, numpy to tensor
        img_gt, img_lq, img_gd = img2tensor([img_gt, img_lq, img_gd], bgr2rgb=False, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        if self.mean_gd is not None or self.std_gd is not None:
            normalize(img_gd, self.mean_gd, self.std_gd, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gd': img_gd, 'lq_path': lq_path, 'gt_path': gt_path, 'gd_path': gd_path}

    def __len__(self):
        return len(self.paths)