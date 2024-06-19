import random
import math
import monai

import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import time
from torch.utils.data import Dataset
import os
import torch
from monai.transforms import Compose, RandAffined, RandAdjustContrastd, RandGaussianSmoothd, RandFlipd, \
    RandScaleIntensityd, RandRotate90d


# 基础数据类，用于读取自定义的数据集
class Getfile(Dataset):
    def __init__(self, base_dir, data_dir, num_classes, label_intensities, mode, onehot, num_data, aug):
        self._base_dir = base_dir
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_intensities = label_intensities
        self.mode = mode
        self.onehot = onehot
        self.image_list = os.listdir(os.path.join(self._base_dir, self.data_dir))
        self.num_data = num_data
        self.aug = aug

        # 数据增强方法示例已给出，请完善，要求使用MONAI框架实现：随机仿射变换、随机对比度调整、随机高斯模糊、随机旋转90度、随机翻转、随机缩放强度
        self.transform = Compose([
            # eg:
            # 随机仿射变换
            RandAffined(keys=['image', 'label'], prob=0.8, rotate_range=(0, 30),
                        scale_range=(0.7, 1.3),
                        mode='bilinear'),
            ############################################################################################################
            # 随机对比度调整
            RandAdjustContrastd(
                keys=['image'], prob=0.8, gamma=(0.5, 1.5)
            ),
            # 随机高斯模糊
            RandGaussianSmoothd(
                keys=['image'], prob=0.8, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)
            ),
            # 随机旋转90度
            RandRotate90d(
                keys=['image', 'label'], prob=0.8, spatial_axes=(0, 1)
            ),
            # 随机翻转
            RandFlipd(
                keys=['image', 'label'], prob=0.8, spatial_axis=0
            ),
            # 随机缩放强度
            RandScaleIntensityd(
                keys=['image'], factors=0.1, prob=0.8
            )
            ############################################################################################################
        ])

    # 数据读取函数，读取npz格式的数据
    def load_npz(self, file_path):
        data = np.load(file_path)
        data_vol = torch.from_numpy(data['image.npy'].astype(np.float32)).float()
        label_vol = torch.from_numpy(data['label.npy'].astype(np.float32)).float()
        return data_vol, label_vol

    def __len__(self):
        if self.num_data == 0:
            return len(self.image_list)
        else:
            return self.num_data

    def __getitem__(self, idx):
        if idx < len(self.image_list) or self.num_data == 0:
            file_name = self.image_list[idx]
        else:
            file_name = random.choice(self.image_list[10:])  # 从第10个索引开始随机选择原图像，用于数据增强

        train_path = os.path.join(self._base_dir, self.data_dir)
        image_path = os.path.join(train_path, file_name)
        image, label = self.load_npz(image_path)

        sample = {'image': image, 'label': label}
        if self.aug and idx >= len(self.image_list):  # 数据增强
            sample = self.transform({'image': image, 'label': label})
        if self.onehot:  # 独热处理
            sample['label'] = get_one_hot_label(sample['label'], self.num_classes,
                                                label_intensities=self.label_intensities,
                                                new_channel=False)
        return sample

    def get_filename(self, idx):
        return self.image_list[idx]


def get_one_hot_label(gt, num_classes, label_intensities=None, new_channel=False):
    if label_intensities is None:
        label_intensities = sorted(torch.unique(gt))
    label = torch.round(gt).to(torch.long)
    if new_channel:
        label = torch.zeros((num_classes, *label.shape), dtype=torch.float32)
        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])
        label[0] = ~torch.sum(label[1:], dim=0).bool()

    else:
        label = torch.zeros((num_classes, *label.shape[1:]), dtype=torch.float32)
        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])
        label[0] = ~torch.sum(label[1:], dim=0).bool()

    return label
