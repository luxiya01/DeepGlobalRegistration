import argparse
import copy
import os
import re
import sys
import json
import logging
import torch
from easydict import EasyDict as edict
from core.trainer import WeightedProcrustesTrainer
from dataloader.base_loader import CollationFunctionFactory

from mbes_data.lib.utils import load_config, setup_seed
from mbes_data.datasets import mbes_data
import wandb
setup_seed(0)

def train(config, resume=False):
  config.dataset_type = 'multibeam_npy_for_dgr'
  name=f'{config.out_dir}'
  run = wandb.init(project='mbes-DGR', name=name,
                   config=config)
  wandb.tensorboard.patch(root_logdir=name)
  train_set, val_set, test_set = mbes_data.get_multibeam_datasets(config)
  collate_pair_fn = CollationFunctionFactory(concat_correspondences=False,
                                          collation_type='collate_pair')
  train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    collate_fn=collate_pair_fn,
    pin_memory=True,
    drop_last=True)
  val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=config.val_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    collate_fn=collate_pair_fn,
    pin_memory=True,
    drop_last=True)

  trainer = WeightedProcrustesTrainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
  )
  trainer.train()
  run.finish()


if __name__ == '__main__':

  ch = logging.StreamHandler(sys.stdout)
  logging.getLogger().setLevel(logging.INFO)
  logging.basicConfig(format='%(asctime)s %(message)s',
                      datefmt='%m/%d %H:%M:%S',
                      handlers=[ch])
  logging.basicConfig(level=logging.INFO, format="")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--mbes_config',
      type=str,
      default='mbes_data/configs/mbes_crop_train_tmp.yaml',
      help='Path to multibeam data config file')
  parser.add_argument(
      '--network_config',
      type=str,
      default='network_configs/kitti.yaml',
      help='Path to network config file')
  args = parser.parse_args()
  mbes_config = edict(load_config(args.mbes_config))
  network_config = edict(load_config(args.network_config))
  config = copy.deepcopy(mbes_config)
  for k, v in network_config.items():
    if k not in config:
      config[k] = v
  os.makedirs(config.exp_dir, exist_ok=True)

  # Resume from checkpoint if available
  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]

    # Get last checkpoint
    pattern = re.compile(r'checkpoint_(\d+)\.pth')
    checkpoints = os.listdir(resume_config['out_dir'])
    latest_epoch = max([int(pattern.match(filename).group(1))
                        for filename in checkpoints if pattern.match(filename)])
    dconfig['resume'] = resume_config['out_dir'] + f'/checkpoint_{latest_epoch}.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  print(config)
  train(config)
