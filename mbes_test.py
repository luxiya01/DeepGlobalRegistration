from collections import defaultdict
import copy
import json
import torch
import logging
import numpy as np
import argparse
import open3d as o3d
import sys
import os
from easydict import EasyDict as edict
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.deep_global_registration import DeepGlobalRegistration

from dataloader.mbesdata_loader import get_multibeam_datasets, get_multibeam_loader
from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.lib.benchmark_utils import to_o3d_pcd, to_tsfm
from mbes_data.lib.evaluations import save_results_to_file, update_results
setup_seed(0)


def draw_results(data, pred_trans):
    gt_trans = to_tsfm(data['transform_gt_rot'], data['transform_gt_trans'])

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(data['points_src'])

    src_pcd_trans = to_o3d_pcd(data['points_src'])
    src_pcd_trans.transform(pred_trans)
    print(f'pred transform: {pred_trans}')

    src_pcd_gt = to_o3d_pcd(data['points_src'])
    src_pcd_gt.transform(gt_trans)
    print(f'gt trans: {gt_trans}')

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(data['points_ref'])

    src_pcd_trans.paint_uniform_color([1, 0, 0])
    src_pcd_gt.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries(
        [src_pcd, ref_pcd, src_pcd_trans, src_pcd_gt])


def test(config):
    logger = logging.getLogger()
    # Load data
    _, _, test_set = get_multibeam_datasets(config)
    test_loader = get_multibeam_loader(config, test_set, shuffle=False)

    dgr = DeepGlobalRegistration(config, device=config.device)

    outdir = os.path.join(config.exp_dir, config.weights)
    os.makedirs(outdir, exist_ok=True)
    results = defaultdict(dict)

    for _, data in tqdm(enumerate(test_loader), total=len(test_set)):
        xyz0, xyz1 = data['pcd0'][0], data['pcd1'][0]
        T_gt = data['T_gt'][0].numpy()
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()
        pred_trans, success = dgr.register(xyz0np, xyz1np)

        eval_data = data['extra_packages'][0]
        eval_data['success'] = success
        results = update_results(results, eval_data, pred_trans,
                       config, outdir, logger)

        if config.draw_registration_results:
            draw_results(eval_data, pred_trans)

    # save results from the last MBES file
    save_results_to_file(logger, results, config, outdir)

if __name__ == '__main__':
    # Set up logging
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d %H:%M:%S',
                        handlers=[ch])
    logging.basicConfig(level=logging.INFO, format="")

    # Load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--mbes_config',
                        type=str,
                        default='mbes_data/configs/mbesdata_test_meters.yaml',
                        help='Path to multibeam data config file')
    parser.add_argument('--network_config',
                        type=str,
                        default='network_configs/kitti.yaml',
                        help='Path to network config file')
    args = parser.parse_args()
    mbes_config = edict(load_config(args.mbes_config))
    network_config = edict(load_config(args.network_config))
    if network_config.resume_dir:
        resume_config = json.load(
            open(network_config.resume_dir + '/config.json', 'r'))
        for k in network_config:
            if k not in ['resume_dir'] and k in resume_config:
                network_config[k] = resume_config[k]
            network_config[
                'resume'] = resume_config['out_dir'] + '/checkpoint.pth'
    print(f'MBES data config: {mbes_config}')
    print(f'Network config: {network_config}')

    config = copy.deepcopy(mbes_config)
    for k, v in network_config.items():
        if k not in config:
            config[k] = v
    if config.use_gpu:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    os.makedirs(config.exp_dir, exist_ok=True)

    test(config)
