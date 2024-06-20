# Unet 项目
欢迎来到 Unet 项目！这是一个简洁而直观的 README 文件，用于指导您如何使用我们的项目。
## 项目概述
本项目是一个调整后的Unet框架。该项目旨在作为医学图像域适应分割的基本框架，在单模态训练下具有较为优越的性能，同时能适应医学图像数据稀缺的问题。
## 特点

- 抗过拟合能力：可在原始数据量少于600的情况下取得较好的分割效果
- 良好的可扩展性：提供了灵活的数据预处理框架，支持MONAI框架下的大部分数据增强操作
- 轻量化：在保证性能的前提下尽可能减少了网络层数和特征通道数，训练和测试高效快速
## 1. [📌重要]环境安装

首先，创建一个新的环境并安装 requirements:
```shell
conda create -n unet python=3.10
conda activate unet
cd Unet/
conda install --yes --file requirements.txt
```

## 2.数据准备

首先，从下面的网址下载数据集：
- **MMWHS 全心分割数据集**: https://github.com/cchen-cc/SIFA#readme. 🚀🚀🚀 **或从该网址获取预处理的数据 [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/Evzk4w-LpoVFgKwa9dwl38EBR_szwDKITwJE0nOue1pLvw?e=joo4ei).**
- **Abdominal**腹部数据集：[CHAOS 医学影像数据集](https://hyper.ai/datasets/20546)  ||[Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge - syn3193805 - Wiki (synapse.org)](https://www.synapse.org/Synapse:syn3193805/wiki/217789)

文件结构应该如下: 
```shell
.
├── data2D
│   ├── MMWHS2D
│   │   ├── CT
│   │   │   ├── ct_train
│   │   │   │   ├── ct_train_1001_image_0.npz
│   │   │   │   └── ...
│   │   │   ├── ct_val
│   │   │   │   ├── ct_val_1006_image_0.npz
│   │   │   │   └── ...
│   │   │   ├── ct_test
│   │   │   │   ├── ct_test_1009_image_0.npz
│   │   │   │   └── ...
│   │   └── MR
│   │       ├── mr_train
│   │       ├── mr_val
│   │       └── mr_test
│   └── Abdominal
│       ├── CT
│       │   ├── ct_train
│       │   │   ├── ct_train_01_image_0.npz
│       │   │   └── ...
│       │   ├── ct_val
│       │   │   ├── ct_val_01_image_0.npz
│       │   │   └── ...
│       │   ├── ct_test
│       │   │   ├── ct_test_01_image_0.npz
│       │   │   └── ...
│       └── MR
│           ├── mr_train
│           ├── mr_val
│           └── mr_test
```

## 使用方法

- **快速开始**: 运行以下命令开始使用 Unet:
```python
# 训练
python train.py --mode CT --gpu 2 -c 5
"""
--mode 训练集模态：[CT,MR,ABCT,ABMR]
--gpu 训练指定显卡
-c 分割类别（包含背景）
"""
# 测试
python test.py --test_mode MR --train_mode CT --gpu 0 --checkpoint 177 178 179
"""
--test_mode 测试集模态：[CT,MR,ABCT,ABMR]
--train_mode 训练集模态：[CT,MR,ABCT,ABMR]
--gpu 训练指定显卡
--checkpoint 要测试的权重文件epoch序号
"""
```

## 贡献
我们欢迎任何形式的贡献，包括但不限于：

- 代码提交
- 问题报告
- 文档改进

## 文档
项目文档可以在 [文档链接] 找到。
## 许可
本项目采用MIT许可证，详情参见**LICENSE 文件**。
## 联系我们

- **问题和帮助**: 如果您有任何问题或需要帮助，请提交一个 GitHub Issue。
- **讨论**: 我们使用Discussions来讨论更广泛的话题。
- **邮件**: 您可以通过邮件联系我们：zlinkw@qq.com。
## 致谢
我们的代码是从[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)修改而来，感谢这些作者的出色工作，希望我们的代码也能推动相关领域的发展



