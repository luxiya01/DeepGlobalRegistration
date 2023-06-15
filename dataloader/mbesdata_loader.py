import argparse
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from dataloader.base_loader import CollationFunctionFactory
from dataloader.inf_sampler import InfSampler
from mbes_data.datasets.mbes_data import MultibeamNpy, get_transforms


def get_multibeam_loader(args: argparse.Namespace,
                         dataset: Dataset,
                         shuffle: bool = False):
    collation_fn = CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair')

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=collation_fn,
                                         num_workers=args.num_workers)
    return loader


def get_multibeam_datasets(args: argparse.Namespace):
    if (args.dataset == 'multibeam'):
        train_set, val_set = get_multibeam_train_datasets(args)
        test_set = get_multibeam_test_datasets(args)
    else:
        raise NotImplementedError
    return train_set, val_set, test_set


def get_multibeam_train_datasets(args: argparse.Namespace):
    train_transforms, val_transforms = get_transforms(
        args.noise_type, args.rot_mag_z, args.trans_mag_x, args.trans_mag_y,
        args.trans_mag_z, args.scale_x, args.scale_y, args.scale_z,
        args.clip_x, args.clip_y, args.clip_z, args.num_points, args.partial)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'multibeam_npy':
        train_data = MBESDataset(args,
                                 args.root,
                                 subset=args.subset_train,
                                 transform=train_transforms)
        val_data = MBESDataset(args,
                               args.root,
                               subset=args.subset_val,
                               transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_multibeam_test_datasets(args: argparse.Namespace):
    _, test_transforms = get_transforms(args.noise_type, args.rot_mag_z,
                                        args.trans_mag_x, args.trans_mag_y,
                                        args.trans_mag_z, args.scale_x,
                                        args.scale_y, args.scale_z,
                                        args.clip_x, args.clip_y, args.clip_z,
                                        args.num_points, args.partial)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'multibeam_npy':
        test_data = MBESDataset(args,
                                args.root,
                                subset=args.subset_test,
                                transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


class MBESDataset(MultibeamNpy):

    def __init__(self, args, root: str, subset: str = 'train', transform=None):
        super().__init__(args, root, subset, transform)

    def __getitem__(self, item):
        data = super().__getitem__(item)

        # Convert to DGR format
        unique_xyz0_th = torch.from_numpy(data['points_src'])
        unique_xyz1_th = torch.from_numpy(data['points_ref'])
        coords0 = torch.floor(unique_xyz0_th / self.config.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.config.voxel_size)
        feats0 = torch.from_numpy(data['features_src'])
        feats1 = torch.from_numpy(data['features_ref'])
        matches = data['matching_inds']
        trans = torch.from_numpy(data['transform_gt'])
        extra_package = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                extra_package[k] = torch.from_numpy(v).float()
            else:
                extra_package[k] = v

        return (unique_xyz0_th.float(), unique_xyz1_th.float(), coords0.int(),
                coords1.int(), feats0.float(), feats1.float(), matches.int(),
                trans, extra_package)
