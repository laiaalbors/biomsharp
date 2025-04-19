import logging
import torch
import pandas as pd
from os import path as osp

import biomsharp.archs
import biomsharp.data
import biomsharp.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_global_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        if dataset_opt["type"] != "GlobalBiomassDataset":
            raise ValueError(f"Global evaluation can only work with GlobalBiomassDataset!")
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        results = model.global_validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        # save results in a CSV file
        csv_results = opt['val']['csv_results']
        csv_results_filename = test_set_name + "_" + csv_results.split('/')[-1]
        csv_results = "/".join(csv_results.split('/')[:-1]) + '/' + csv_results_filename
        df_results = pd.DataFrame.from_dict(results, orient='index')
        # Reset index to make the file path a column
        df_results.reset_index(inplace=True)
        df_results.rename(columns={'index': 'file_path'}, inplace=True)
        # Save the DataFrame to a CSV file
        df_results.to_csv(csv_results, index=False)
        print(f"Evaluation saved in {csv_results}.")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_global_pipeline(root_path)