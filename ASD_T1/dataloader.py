import pandas as pd
import numpy as np
import scipy.io as scio
from monai.data import DataLoader, Dataset, partition_dataset
import torch
import torch.nn as nn
import Models
from monai.transforms import (
    EnsureChannelFirstd,
    AddChanneld,
    Compose,
    LoadImaged,
    SaveImaged,
    ScaleIntensityd,
    SpatialCropd,
    SpatialPadd,
    RandFlipd,
    EnsureTyped, RandRotated, RandZoomd,
)
import os
import json


class ABIDE1_fMRI_dataset():
    def __init__(self,cv):
        self.subInfo = pd.read_csv('/public_bme/share/sMRI/Yuanbo//subinfo_ABIDE.csv')
        temp = pd.read_csv('/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv')
        self.index = temp['index']
        self.y_data1 = torch.from_numpy(self.subInfo['disease'].to_numpy())
        self.y_data1 = torch.tensor(self.y_data1, dtype=torch.float32)
        self.y_data1 = self.y_data1[self.index]
        self.index = self.index.tolist()

    def load_data(self):
        data_list = []
        for id, label in zip(self.index, self.y_data1):
            fn = self.subInfo['filename'][id]
            fn = fn + '.mat'
            FCfile = scio.loadmat('/public_bme/share/sMRI/Yuanbo/FC/par500/' + fn)
            FC = FCfile['FC']
            FC = torch.tensor(FC, dtype=torch.float32)
            data_dict = {'fMRI': FC, 'label':label.item()}
            data_list.append(data_dict)
        return data_list

class ABIDE1_T1_dataset():
    def __init__(self,cv):
        self.subInfo = pd.read_csv('/public_bme/share/sMRI/Yuanbo//subinfo_ABIDE.csv')
        temp = pd.read_csv('/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv')
        self.index = temp['index']
        self.y_data1 = torch.from_numpy(self.subInfo['disease'].to_numpy())
        self.y_data1 = torch.tensor(self.y_data1,dtype=torch.int64)
        self.y_data1 = self.y_data1[self.index]
        self.index = self.index.tolist()
        self.labels = self.y_data1.tolist()
        self.data_list = []

    def load_data(self):
        data_T1 = []
        labels = self.labels
        for id in self.index:
            fn = self.subInfo['filename'][id]
            T1_adress = '/public_bme/share/sMRI/Yuanbo/T1_RIGID/ABIDE/' + fn + '/Warped.nii.gz'
            data_T1.append(T1_adress)
        data_list = [{"T1": img, "label": label} for img, label in zip(data_T1, labels)]
        train_transform, test_transform = transform()
        train_ds = Dataset(data=data_list, transform=train_transform)
        test_ds = Dataset(data=data_list, transform=test_transform)
        return data_list

    def __len__(self):
        return len(self.labels)

    def get_weights(self):
        label_list = []
        for item in self.data_list:
            label_list.append(item['label'])
        return float(label_list.count(0)), float(label_list.count(1))

def get_T1_dataloader():
    print('----------------- Dataset -------------------')
    print('Loading ABIDE1 T1 dataset...')
    abide1_dataset = ABIDE1_T1_dataset(1)
    data_list = abide1_dataset.load_data()
    ABIDE1_T1_partitions = partition_dataset(data=data_list, ratios=[0.8, 0.2], shuffle=True)
    train_dataset, val_dataset = ABIDE1_T1_partitions[0], ABIDE1_T1_partitions[1]
    train_transform, test_transform = transform()
    train_dataset = Dataset(train_dataset, transform=train_transform)
    val_dataset = Dataset(val_dataset, transform=test_transform)
    print('The number of training images = %d' % len(train_dataset))
    print('The number of val images = %d' % len(val_dataset))

    num_workers = 0

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers= num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=8, num_workers= num_workers)
    # asd_sum = sum(1 for item in train_dataset if item['label'] == 1)
    # nc_sum = len(train_dataset) - asd_sum
    # ratio = asd_sum / nc_sum
    # asd_sum_test = sum(1 for item in val_dataset if item['label'] == 1)
    # nc_sum_test = len(val_dataset) - asd_sum_test
    # ratio1 = asd_sum_test / nc_sum_test

    # return train_dataloader, val_dataloader, ratio, ratio1
    return train_dataloader, val_dataloader

class ABIDE1_dataset():
    def __init__(self, cv):
        self.subInfo = pd.read_csv('/public_bme/share/sMRI/Yuanbo/subinfo_ABIDE.csv')
        temp = pd.read_csv('/public_bme/home/wangyb12023/shuffled_index' + str(cv) + '.csv')
        self.index = temp['index']
        self.y_data1 = torch.from_numpy(self.subInfo['disease'].to_numpy())
        self.y_data1 = torch.tensor(self.y_data1, dtype=torch.float32)
        self.y_data1 = self.y_data1[self.index]
        self.index = self.index.tolist()

    def load_data(self):
        data_list = []
        for id, label in zip(self.index, self.y_data1):
            fn = self.subInfo['filename'][id]
            t1_path = '/public_bme/share/sMRI' + fn + '/Warped.nii.gz'
            fn = fn + '.mat'
            fc_file = scio.loadmat('/public_bme/share/sMRI/Yuanbo/FC/par500/' + fn)
            fc_data = torch.tensor(fc_file['FC'], dtype=torch.float32)
            data_dict = {'T1': t1_path, 'fMRI': fc_data, 'label': label.item()}
            data_list.append(data_dict)
        return data_list
# class ABIDE_fMRI_Dataset(Dataset):
#     def __init__(self,cv):
#         super(ABIDE_fMRI_Dataset, self).__init__()
#         self.cv = cv
#         self.FC = []
#         self.y_data = []
#         self.FC, self.y_data = ABIDE1_fMRI_dataset(cv)
#
#     def __getitem__(self, idx):
#         FC = self.FC[idx]
#         labels = self.y_data[idx]
#         return FC, labels
#
#     def __len__(self):
#         return len(self.y_data)




class ABIDE1_Dataset(Dataset):
    def __init__(self, data_list, data_type='T1'):
        # super(ABIDE1_Dataset,self).__init__()
        self.data_list = data_list
        self.data_type = data_type

        # Define transformations based on data type
        if self.data_type == 'T1':
            self.transform = Compose([
                LoadImaged(keys=['T1']),
                EnsureChannelFirstd(keys=['T1'],channel_dim="no_channel"),
                ScaleIntensityd(keys=['T1']),
                EnsureTyped(keys=['T1'])
            ])
        elif self.data_type == 'fMRI':
             self.transform = None  # No transformation needed for fMRI
        elif self.data_type == 'T1 and fMRI':
            # Define transformation for T1 images
             self.transform_T1 = Compose([
                LoadImaged(keys=['T1']),
                EnsureChannelFirstd(keys=['T1'],channel_dim="no_channel"),
                ScaleIntensityd(keys=['T1']),
                EnsureTyped(keys=['T1'])
             ])
            # No transformation needed for fMRI
             self.transform_fMRI = None
        else:
             raise ValueError("Invalid data_type. Please specify 'T1', 'fMRI', or 'T1 and fMRI'.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        label = data_item.get('label', None)  # 获取标签

        # Apply transformation based on data type
        if self.data_type == 'T1':
            data_item_T1 = self.transform(data_item['T1'])
            return {'T1': data_item_T1['T1'], 'label': label}  # 返回包含 T1 数据和标签的字典
        elif self.data_type == 'fMRI':
            # No transformation needed for fMRI
            return {'fMRI': data_item['fMRI'], 'label': label}  # 返回包含 fMRI 数据和标签的字典
        elif self.data_type == 'T1 and fMRI':
            data_item_T1 = self.transform_T1(data_item['T1'])
            data_item_fMRI = data_item['fMRI']
            return {'T1': data_item_T1['T1'], 'fMRI': data_item_fMRI, 'label': label}  # 返回包含 T1、fMRI 数据和标签的字典
        else:
            raise ValueError("Invalid data_type. Please specify 'T1', 'fMRI', or 'T1 and fMRI'.")






def transform():
    train_transform = Compose([
                LoadImaged(keys=['T1']),
                # EnsureChannelFirstd(keys=['T1'], channel_dim=0),
                AddChanneld(keys=['T1']),
                RandFlipd(keys=['T1'], prob=0.5, spatial_axis=1),
                RandRotated(keys=['T1'], prob=0.5, range_x=(-0.09, 0.09), range_y=(-0.09, 0.09), range_z=(-0.09, 0.09), mode="bilinear"),
                RandZoomd(keys=['T1'], prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear"),
                ScaleIntensityd(keys=['T1']),
                EnsureTyped(keys=['T1'])
    ])
    test_transform = Compose([
                LoadImaged(keys=['T1']),
                # EnsureChannelFirstd(keys=['T1'], channel_dim=0),
                AddChanneld(keys=['T1']),
                RandFlipd(keys=['T1'], prob=0.5, spatial_axis=1),
                RandRotated(keys=['T1'], prob=0.5,range_x=(-0.09, 0.09), range_y=(-0.09, 0.09), range_z=(-0.09, 0.09), mode="bilinear"),
                RandZoomd(keys=['T1'], prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear"),
                ScaleIntensityd(keys=['T1']),
                EnsureTyped(keys=['T1'])
    ])
    return train_transform, test_transform


# def transform():
#     train_transform = Compose([
#                 LoadImaged(keys=['T1']),
#                 EnsureChannelFirstd(keys=['T1'],channel_dim=1),
#                 ScaleIntensityd(keys=['T1']),
#                 EnsureTyped(keys=['T1'])
#     ])
#     test_transform = Compose([
#                 LoadImaged(keys=['T1']),
#                 EnsureChannelFirstd(keys=['T1'],channel_dim=1),
#                 ScaleIntensityd(keys=['T1']),
#                 EnsureTyped(keys=['T1'])
#     ])
#     return train_transform, test_transform

def save_dataset_partition(train_dataset, val_dataset, test_dataset, save_dir):
    partition_dict = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    save_path = os.path.join(save_dir, 'partition.json')
    with open(save_path, 'w') as f:
        json.dump(partition_dict, f)

def get_dataloader(data_type='T1'):
    if data_type == 'T1':
        print('----------------- Dataset -------------------')
        print('Loading ABIDE1 T1 dataset...')
        abide1_dataset = ABIDE1_T1_dataset(1)
        data_list = abide1_dataset.load_data()
        ABIDE1_T1_partitions = partition_dataset(data=data_list, ratios=[0.8, 0.2], shuffle=True)
        train_dataset, val_dataset = ABIDE1_T1_partitions[0], ABIDE1_T1_partitions[1]
        train_dataset = ABIDE1_Dataset(train_dataset, data_type='T1')
        val_dataset = ABIDE1_Dataset(val_dataset, data_type='T1')
        print('The number of training images = %d' % len(train_dataset))
        print('The number of val images = %d' % len(val_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=8)
        asd_sum = sum(1 for item in train_dataset if item['label'] == 1)
        nc_sum = len(train_dataset) - asd_sum
        ratio = asd_sum / nc_sum
        asd_sum_test = sum(1 for item in val_dataset if item['label'] == 1)
        nc_sum_test = len(val_dataset) - asd_sum_test
        ratio1 = asd_sum_test / nc_sum_test
    elif data_type == 'fMRI':
        print('----------------- Dataset -------------------')
        print('Loading ABIDE1 fMRI dataset...')
        abide1_dataset = ABIDE1_fMRI_dataset(1)
        data_list = abide1_dataset.load_data()
        ABIDE1_fMRI_partitions = partition_dataset(data=data_list, ratios=[0.8, 0.2], shuffle=True)
        train_dataset, val_dataset = ABIDE1_fMRI_partitions[0], ABIDE1_fMRI_partitions[1]
        train_dataset = ABIDE1_Dataset(train_dataset, data_type='fMRI')
        val_dataset = ABIDE1_Dataset(val_dataset, data_type='fMRI')
        print('The number of training images = %d' % len(train_dataset))
        print('The number of val images = %d' % len(val_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=30)
        asd_sum = sum(1 for item in train_dataset if item['label'] == 1)
        nc_sum = len(train_dataset) - asd_sum
        ratio = asd_sum / nc_sum
        asd_sum_test = sum(1 for item in val_dataset if item['label'] == 1)
        nc_sum_test = len(val_dataset) - asd_sum_test
        ratio1 = asd_sum_test / nc_sum_test
    elif data_type == 'T1 and fMRI':
        print('----------------- Dataset -------------------')
        print('Loading ABIDE1 T1 and fMRI dataset...')
        abide1_dataset = ABIDE1_dataset(1)
        data_list = abide1_dataset.load_data()
        ABIDE1_partitions = partition_dataset(data=data_list, ratios=[0.8, 0.2], shuffle=True)
        train_dataset, val_dataset = ABIDE1_partitions[0], ABIDE1_partitions[1]
        train_dataset = ABIDE1_Dataset(train_dataset, data_type='T1 and fMRI')
        val_dataset = ABIDE1_Dataset(val_dataset, data_type='T1 and fMRI')
        print('The number of training images = %d' % len(train_dataset))
        print('The number of val images = %d' % len(val_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=30)
        asd_sum = sum(1 for item in train_dataset if item['label'] == 1)
        nc_sum = len(train_dataset) - asd_sum
        ratio = asd_sum / nc_sum
        asd_sum_test = sum(1 for item in val_dataset if item['label'] == 1)
        nc_sum_test = len(val_dataset) - asd_sum_test
        ratio1 = asd_sum_test / nc_sum_test

    return train_dataloader, val_dataloader, ratio, ratio1

def ratio_calculator():
    ratio = ABIDE1_dataset.y_data1[0:cut].sum() / (y_data1[0:cut].shape[0] - y_data1[0:cut].sum())
    if ratio < 1:
        weight = torch.cuda.FloatTensor([1, 1 / ratio])
    else:
        weight = torch.cuda.FloatTensor([ratio, 1])



















