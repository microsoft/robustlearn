# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data_util.sensor_loader import SensorDataset
from torch.utils.data import DataLoader
from data_util.fast_data_loader import InfiniteDataLoader
import numpy as np


def load(args):
    """Load data and get dataloader
    """
    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = [], [], [], [], [], []
    data_path = args.root_path + \
        f'{args.dataset}_crosssubject_rawaug_rate{args.remain_data_rate}_t{args.target}_seed{args.seed}_scaler{args.scaler_method}.pkl'
    data_raw_aug = np.load(data_path, allow_pickle=True)
    raw_trs, aug_trs, raw_vas, aug_vas, raw_tet, aug_tet = data_raw_aug['raw_trs'], data_raw_aug[
        'aug_trs'], data_raw_aug['raw_vas'], data_raw_aug['aug_vas'], data_raw_aug['raw_tet'], data_raw_aug['aug_tet']

    train_raw_dataset = SensorDataset(
        raw_trs, aug=False, dataset=args.dataset)
    train_aug_dataset = SensorDataset(
        aug_trs, aug=True, dataset=args.dataset)

    val_raw_dataset = SensorDataset(
        raw_vas, aug=False, dataset=args.dataset)
    val_aug_dataset = SensorDataset(
        aug_vas, aug=True, dataset=args.dataset)

    test_raw_dataset = SensorDataset(
        raw_tet, aug=False, dataset=args.dataset)
    test_aug_dataset = SensorDataset(
        aug_tet, aug=True, dataset=args.dataset)

    tstep_per_epoch = int(len(aug_trs[0])/args.batch_size)
    if tstep_per_epoch < args.step_per_epoch:
        args.step_per_epoch = tstep_per_epoch

    train_raw_loader = InfiniteDataLoader(
        dataset=train_raw_dataset,
        sample_weights=None,
        batch_size=args.batch_size//2,
        num_workers=args.num_workers)

    train_aug_loader = InfiniteDataLoader(
        dataset=train_aug_dataset,
        sample_weights=None,
        batch_size=args.batch_size//2,
        num_workers=args.num_workers)

    val_raw_loader = DataLoader(
        dataset=val_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    val_aug_loader = DataLoader(
        dataset=val_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_raw_loader = DataLoader(
        dataset=test_raw_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    test_aug_loader = DataLoader(
        dataset=test_aug_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)

    return train_raw_loader, train_aug_loader, val_raw_loader, val_aug_loader, test_raw_loader, test_aug_loader
