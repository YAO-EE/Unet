import argparse
import configparser
import copy
import json
import logging
import os
import sys
import time
from ast import literal_eval

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.dataloader import Getfile
from val import calculate_metric_percase
from model.unet import UNet2D

parser = argparse.ArgumentParser(description='Testing the UNet')
parser.add_argument('--data_path', type=str,
                    default='../data2D', help='Name of Experiment')
parser.add_argument('--test_save_path', type=str,
                    default='predicatedimg',
                    help='Name of Experiment')
parser.add_argument('--checkpoint_path', type=str,
                    default='checkpoints',
                    help='Name of Experiment')
parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=20, help='Batch size')
parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
parser.add_argument('--train_mode', type=str, default="ABMR", help='train mode')
parser.add_argument('--test_mode', type=str, default="ABCT", help='test mode to run')
parser.add_argument('--gpu', type=int, default=4, help='gpu to run')
parser.add_argument('--checkpoint', nargs='+', type=int, default=[0], help='checkpoints to run')
args = parser.parse_args()

"""
python test.py --test_mode MR --train_mode CT --gpu 0 --checkpoint 177 178 179

"""


def test(args):
    data_path = args.data_path
    test_save_path = args.test_save_path
    os.makedirs(test_save_path, exist_ok=True)
    batch_size = args.batch_size
    num_classes = args.classes
    checkpoint_path = args.checkpoint_path  # 指定 checkpoint 文件路径
    train_mode = args.train_mode
    test_mode = args.test_mode
    checkpoint_path = os.path.join(checkpoint_path, train_mode + "2D")
    checkpoint = args.checkpoint

    config = configparser.ConfigParser()
    print("Loading config.ini")
    config.read('config2D.ini')  # 替换为配置文件路径
    test_dir_list = config.get(test_mode, 'test_dir')
    test_dir = test_dir_list.split(", ")
    label_intensities_str = config.get(test_mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(',')))
    class_to_pixel_str = config.get(test_mode, 'class_to_pixel')
    class_to_pixel = json.loads(class_to_pixel_str)

    ####################################################################################################################
    test_dataset = Getfile(base_dir=data_path,
                           data_dir=test_dir[0], num_classes=num_classes,
                           label_intensities=label_intensities, mode=test_mode, onehot=False, num_data=0, aug=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, drop_last=True)
    print('Loading data... ', test_dir[0])
    # 初始化模型
    net = UNet2D(in_channels=3, out_channels=args.classes)
    net.to(device=device)

    print("Testing begin")
    # 支持测试多个检查点，方便对比
    for epoch in checkpoint:
        # 加载权重数据
        net = load_checkpoint(net, checkpoint_path, f"checkpoint_epoch{epoch}.pth")
        net.eval()

        sum_dice_scores = np.zeros(num_classes - 1)
        non_zero_counts = np.zeros(num_classes - 1)

        for idx, batch in enumerate(test_loader):
            images, labels = batch['image'].cuda(), batch['label'].cuda()
            metrics = test_single_volume(images, labels, net, num_classes, class_to_pixel, test_save_path, idx)
            for c, metric in enumerate(metrics):
                sum_dice_scores[c] += metric
                non_zero_counts[c] += 1

        avg_scores_except_first = np.mean(sum_dice_scores / non_zero_counts)
        avg_dice_scores_per_class = sum_dice_scores / non_zero_counts

        for c in range(0, num_classes - 1):
            logging.info(f"Avg Dice Score for Class {c + 1}: {float(avg_dice_scores_per_class[c])}")
        logging.info(f"Avg Dice Score for Classes (excluding Class 0): {float(avg_scores_except_first)}")
        logging.info("Evaluating end")

        return avg_scores_except_first


def test_single_volume(images, labels, net, num_classes, class_to_pixel, save_path, batch_idx):
    net.eval()
    with torch.no_grad():
        # 对整个batch验证，加快验证速度
        preds = net(images)
        predictions = torch.zeros_like(labels)
        # 获取argmax结果
        for ind in range(images.shape[0]):
            pred = preds[ind].argmax(dim=0)
            out = torch.take(torch.tensor(list(class_to_pixel.values())).cuda(), pred).unsqueeze(0)
            predictions[ind] = out.cpu()
            # 将预测结果和原始图像叠加在一起
            image = images[ind].cpu().numpy().transpose(1, 2, 0)
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            label = labels[ind].cpu().numpy()
            pred = out[0].cpu().numpy()

            overlay = image.copy()
            overlay[pred > 0] = [255, 0, 0]

            # 可视化和保存叠加结果
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(label, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')

            plt.savefig(os.path.join(save_path, f'batch{batch_idx}_img{ind}_overlay.png'))
            plt.close()

    metric_list = [calculate_metric_percase(predictions == i, labels == i) for i in range(1, num_classes)]
    return metric_list


def load_checkpoint(model, checkpoint_path, checkpoint_name):
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_name), map_location=device)
    print('Loading...', os.path.join(checkpoint_path, checkpoint_name))
    model.load_state_dict(checkpoint)
    return model


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 将FileHandler添加到日志记录器
    logger = logging.getLogger()
    # 指定显卡序号
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    test(args)
