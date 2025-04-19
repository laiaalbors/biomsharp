import math
import numpy as np
from tqdm import tqdm
from os import path as osp
from collections import OrderedDict

import rioxarray
from rasterio.warp import Resampling

import torch
from torch.nn import functional as F

from basicsr.models.sr_model import SRModel
from basicsr.utils import imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models import lr_scheduler as lr_scheduler

from biomsharp.metrics import calculate_metric
from biomsharp.utils import tensor2img, imwrite_rasterio, assign_georeference, img2tensor


@MODEL_REGISTRY.register()
class NaiveModel(SRModel):

    def global_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
            sample_results = {metric: 0 for metric in self.metric_results}
            results = {}
        
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        total = len(dataloader)
        for idx, val_data in enumerate(dataloader):
            print(f"    {idx + 1}/{total}", flush=True)

            gd_path = val_data['gt_path'][0]
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            visuals = OrderedDict()
            visuals['result'] = val_data['upscaled_lq']
            visuals['gt'] = val_data['gt']
            sr_img = tensor2img([visuals['result']], out_type=np.uint16)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], out_type=np.uint16)
                metric_data['img2'] = gt_img
            
            # tentative for out of GPU memory
            if idx % 10 == 0:
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.tif')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.tif')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.tif')
                imwrite_rasterio(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_result = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += metric_result
                    sample_results[name] = metric_result
                # results[gd_path] = sample_results.copy()
                print(f"    {gd_path}: {sample_results}", flush=True)
            
            del sr_img
            if 'gt_img' in locals():
                del gt_img

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        
        return results
    

    def france_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
            sample_results = {metric: 0 for metric in self.metric_results}
        
        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        total = len(dataloader)
        for idx, val_data in enumerate(dataloader):
            print(f"    {idx + 1}/{total}", flush=True)

            gd_path = val_data['gt_path'][0]
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            visuals = OrderedDict()
            visuals['result'] = val_data['upscaled_lq']

            output = visuals['result'].squeeze(0).float().detach().cpu().numpy()
            output_25m = assign_georeference(output, val_data['properties'])
            output_25m = output_25m.rio.write_nodata(0)
            gt_30m = rioxarray.open_rasterio(val_data['gt_path'][0])
            output_30m = output_25m.rio.reproject_match(gt_30m, resampling=Resampling.nearest)
            output_30m_np = np.nan_to_num(output_30m.values, copy=False, nan=-1.0).transpose(1, 2, 0)
            output_30m_np = img2tensor(output_30m_np, bgr2rgb=False, float32=True)

            visuals['gt'] = val_data['gt']
            sr_img = tensor2img([output_30m_np], out_type=np.uint16)
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], out_type=np.uint16)
                metric_data['img2'] = gt_img
            
            # tentative for out of GPU memory
            if idx % 10 == 0:
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.tif')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.tif')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.tif')
                imwrite_rasterio(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_result = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += metric_result
                    sample_results[name] = metric_result
                print(f"    {gd_path}: {sample_results}", flush=True)
            
            del sr_img
            if 'gt_img' in locals():
                del gt_img

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
