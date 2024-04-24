from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import nibabel as nib
import os
import torch
import random
from monai.transforms import Compose, EnsureChannelFirst, RandFlip, RandRotate, RandZoom
# import utils
import scipy.io as scio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd


def normalize_norm(img):
    # normalize to 0 - 1
    img = img + 1  # 0 - 2
    min_v = np.min(img)
    max_v = np.max(img)
    ratio = 1.0 * (1 - 0) / (max_v - min_v)
    img = 1.0 * (img - min_v) * ratio
    return img


class ExCustomDataset(Dataset):
    def __init__(self, df, temp, transforms=None, rigid=False, mask_aug=False):
        super(ExCustomDataset, self).__init__()
        self.df = df
        self.temp = temp
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            # RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            # RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])
        # self.rigid = rigid

    def __getitem__(self,index):
        fn = self.df.loc[index, 'filename']

        # ID = self.df.loc[index, 'No']
        # filename = self.df.loc[index, 'filename']
        # site = self.df.loc[index, 'site']

        # use rigid registrated sMRI

        sMRI = nib.load(os.path.join('/public_bme/share/sMRI/Yuanbo/T1_RIGID/ABIDE/', fn, 'Warped.nii.gz'))

        data_sMRI = np.array(sMRI.dataobj)  # or sMRI.get_fdata()

        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)
        # data_sMRI = data_sMRI/data_sMRI.mean()
        labels = torch.tensor(self.df.loc[index, 'disease'], dtype=torch.float32)
        # age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        # sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        # data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        return data_sMRI.unsqueeze(0), labels

    def __len__(self):
        return self.df.shape[0]


class ExCustomDataset_Dual():
    def __init__(self, temp, df=None, transforms=False):
        self.df = df
        self.temp = temp
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            # RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            # RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])

    def __getitem__(self,index):
        fn = self.df.loc[index, 'filename']

        # site = self.df.loc[index, 'site']
        # age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        # sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        sMRI = nib.load(os.path.join('/public_bme/share/sMRI/Yuanbo/T1_RIGID/ABIDE/', fn, 'Warped.nii.gz'))

        data_sMRI = np.array(sMRI.dataobj)  # or sMRI.get_fdata()
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)
        labels = torch.tensor(self.df.loc[index, 'disease'], dtype=torch.float32)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        # data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        # load FC
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par500/', fn , '.mat'))
        FC = FCfile['FC']
        FC = torch.tensor(FC, dtype=torch.float32)

        # FC1 = load_FCN(fMRI_path, filename, 100)
        # FC2 = load_FCN(fMRI_path, filename, 200)
        # FC3 = load_FCN(fMRI_path, filename, 300)
        # FC4 = load_FCN(fMRI_path, filename, 400)
        # FC5 = load_FCN(fMRI_path, fn, 500)
        fMRI = FC
        return data_sMRI.unsqueeze(0), fMRI, labels

    def __len__(self):
        return self.df.shape[0]


class ExCustomDataset_Dual_MAHGCN():
    def __init__(self, temp, df=None, transforms=False):
        self.df = df
        self.temp = temp
        self.transforms = transforms
        self.transforms_op = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            RandFlip(prob=0.5, spatial_axis=1),
            # RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
            # RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        ])

    def __getitem__(self,index):
        fn = self.df.loc[index, 'filename']

        # site = self.df.loc[index, 'site']
        # age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        # sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)

        sMRI = nib.load(os.path.join('/public_bme/share/sMRI/Yuanbo/T1_RIGID/ABIDE/', fn, 'Warped.nii.gz'))

        data_sMRI = np.array(sMRI.dataobj)  # or sMRI.get_fdata()
        data_sMRI = normalize_norm(data_sMRI)
        data_sMRI = torch.tensor(data_sMRI, dtype=torch.float32)
        labels = torch.tensor(self.df.loc[index, 'disease'], dtype=torch.float32)

        if self.transforms:
            data_sMRI = self.transforms_op(data_sMRI)[0]

        # data_sMRI = utils.crop_center(data_sMRI, (160, 192, 160))

        # load FC
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par100/', fn + '.mat'))
        FC1 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par200/', fn + '.mat'))
        FC2 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par300/', fn + '.mat'))
        FC3 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par400/', fn + '.mat'))
        FC4 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par500/', fn + '.mat'))
        FC5 = FCfile['FC']

        FC1 = torch.tensor(FC1, dtype=torch.float32)
        FC2 = torch.tensor(FC2, dtype=torch.float32)
        FC3 = torch.tensor(FC3, dtype=torch.float32)
        FC4 = torch.tensor(FC4, dtype=torch.float32)
        FC5 = torch.tensor(FC5, dtype=torch.float32)
        fMRI = [FC1, FC2, FC3, FC4, FC5]

        return data_sMRI.unsqueeze(0), fMRI, labels

    def __len__(self):
        return self.df.shape[0]

class ExCustomDataset_fMRI():
    def __init__(self, temp, df=None, transforms=False):
        self.df = df
        self.temp = temp
        # self.transforms = transforms
        # self.transforms_op = Compose([
        #     EnsureChannelFirst(channel_dim="no_channel"),
        #     RandFlip(prob=0.5, spatial_axis=1),
        #     # RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
        #     # RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        # ])

    def __getitem__(self,index):
        fn = self.df.loc[index, 'filename']

        # site = self.df.loc[index, 'site']
        # age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        # sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)
        labels = torch.tensor(self.df.loc[index, 'disease'], dtype=torch.float32)


        # load FC
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par500/', fn + '.mat'))
        FC = FCfile['FC']
        FC = torch.tensor(FC, dtype=torch.float32)

        # FC1 = load_FCN(fMRI_path, filename, 100)
        # FC2 = load_FCN(fMRI_path, filename, 200)
        # FC3 = load_FCN(fMRI_path, filename, 300)
        # FC4 = load_FCN(fMRI_path, filename, 400)
        # FC5 = load_FCN(fMRI_path, fn, 500)
        fMRI = FC
        return fMRI, labels

    def __len__(self):
        return self.df.shape[0]

class ExCustomDataset_fMRI_MAHGCN():
    def __init__(self, temp, df=None, transforms=False):
        self.df = df
        self.temp = temp
        # self.transforms = transforms
        # self.transforms_op = Compose([
        #     EnsureChannelFirst(channel_dim="no_channel"),
        #     RandFlip(prob=0.5, spatial_axis=1),
        #     # RandRotate(prob=0.5, range_x=[-0.09, 0.09], range_y=[-0.09, 0.09], range_z=[-0.09, 0.09], mode="bilinear"),
        #     # RandZoom(prob=0.5, min_zoom=[0.95, 0.95, 0.95], max_zoom=[1.05, 1.05, 1.05], mode="trilinear")
        # ])

    def __getitem__(self,index):
        fn = self.df.loc[index, 'filename']

        # site = self.df.loc[index, 'site']
        # age = torch.tensor(self.df.loc[index, 'age'], dtype=torch.float32)
        # sex = torch.tensor(self.df.loc[index, 'sex'], dtype=torch.long)
        labels = torch.tensor(self.df.loc[index, 'disease'], dtype=torch.float32)


        # load FC
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par100/', fn +'.mat'))
        FC1 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par200/', fn +'.mat'))
        FC2 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par300/', fn +'.mat'))
        FC3 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par400/', fn +'.mat'))
        FC4 = FCfile['FC']
        FCfile = scio.loadmat(os.path.join('/public_bme/share/sMRI/Yuanbo/FC/par500/', fn +'.mat'))
        FC5 = FCfile['FC']

        FC1 = torch.tensor(FC1, dtype=torch.float32)
        FC2 = torch.tensor(FC2, dtype=torch.float32)
        FC3 = torch.tensor(FC3, dtype=torch.float32)
        FC4 = torch.tensor(FC4, dtype=torch.float32)
        FC5 = torch.tensor(FC5, dtype=torch.float32)
        fMRI = [FC1, FC2, FC3, FC4, FC5]

        return fMRI, labels

    def __len__(self):
        return self.df.shape[0]


def data_split(dataset=None, seed=None):
    """
    inner dataset split without external validation, train/val/test/ 8:1:1
    """
    n_out_tests = 0
    np.random.seed(seed)

    in_indices = dataset.temp['index'].values.tolist()
    in_L = len(in_indices)
    np.random.shuffle(in_indices)

    n_in_tests = int(np.floor(in_L*0.1))
    n_vals = int(np.floor(in_L*0.1))

    split = n_in_tests + n_vals
    n_trains = in_L - split

    in_test_indices, val_indices, train_indices = in_indices[:n_in_tests], in_indices[n_in_tests: split], in_indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    in_test_sampler = SubsetRandomSampler(in_test_indices)
    out_test_sampler = None
    return train_sampler, val_sampler, in_test_sampler, out_test_sampler, n_trains, n_vals, n_in_tests, n_out_tests

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


# def load_FCN(fMRI_path, filename, ROI):
#     A = np.zeros((ROI, ROI))
#     mat_path = fMRI_path + '/' + 'par' + str(ROI) + '/' + filename + '.mat'
#     FCfile = scio.loadmat(mat_path)
#     A = FCfile['FC']
#
#     # # high order, second order
#     # FC = np.corrcoef(A)
#     np.fill_diagonal(A, 0)  # 1 is better
#     # A = FC
#     # A = torch.tensor(A, dtype=torch.float32)
#
#     A = torch.tensor(A)
#     if ROI not in [400, 500]:
#         tau = 0.0  # weak connectivity removed, noise
#         A = torch.where(A > tau, A, 0.)
#     elif ROI in [400, 500]:
#         tau = 0.3
#         A = torch.where(torch.abs(A) > tau, A, 0.)
#     else:
#         pass
#
#     A = A.type(torch.float32)
#
#     # DropEdge
#     DropEdge = False
#     # if DropEdge: # old random
#     #     sampling_p = 0.05 # sampling probability
#     #     nnz = torch.count_nonzero(A) # n_non_zeros
#     #     preserve_nnz = int(nnz*sampling_p)//2
#     #     triu_arr = torch.triu(A, diagonal=1)
#
#     #     mask = torch.zeros((ROI, ROI))
#     #     mask.bernoulli_(sampling_p)
#     #     mask_triu = torch.triu(mask, diagonal=1)
#     #     mask_tril = mask_triu.t() # transpose
#     #     A_triu = mask_triu*A
#     #     A_tril = mask_tril*A
#     #     A = A_triu + A_tril
#     if DropEdge:
#         top_percentage = 0.2  # Percentage of top edge connections to select
#         nnz = torch.count_nonzero(A)  # Number of non-zero elements
#         preserve_nnz = int(nnz * top_percentage)  # Number of edges to select
#
#         # Flatten the adjacency matrix and get the indices of the top edge connections
#         sorted_indices = torch.argsort(A.flatten(), descending=True)
#
#         # Create a binary mask to select the top edge connections
#         mask = torch.zeros_like(A)
#         mask_flattened = mask.flatten()
#         mask_flattened[sorted_indices[:preserve_nnz]] = 1
#         mask = mask_flattened.reshape(A.shape)
#
#         # Apply the mask to the adjacency matrix to keep only the selected edges
#         A = A * mask
#
#     return A