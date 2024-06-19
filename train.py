import argparse
import configparser
import json
import logging
import random
import sys
import time
from ast import literal_eval
from datetime import timedelta

import matplotlib.pyplot as plt
import monai
import numpy as np
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss
from model.unet import UNet2D
from utils.dataloader import Getfile
from torch.optim.lr_scheduler import ExponentialLR
import os

from val import evaluate

# 参数解析器，用于从命令行读取参数
parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('--data_path', type=str,
                    default='../data2D', help='Path of Data')
parser.add_argument('--checkpoint_path', type=str,
                    default='checkpoints', help='Path of checkpoints')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-4,
                    help='Learning rate', dest='lr')
parser.add_argument('--load', '-f', type=int, default=0, help='Load model from a .pth file')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='save_checkpoint')

parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
parser.add_argument('--seed', type=int, default=8888, help='random seed')
parser.add_argument('--patience', type=int, default=50, help='patience')
parser.add_argument('--model_type', type=str, default='unet2D', choices=['unet', 'unet2D'],
                    help='Type of the model to use')
parser.add_argument('--mode', type=str, default="CT", help='mode to run')
parser.add_argument('--gpu', type=int, default=0, help='gpu to run')
args = parser.parse_args()


#########################################################################
# 这里仅使用其中一种模态训练即可，如CT
# python train.py --mode CT --gpu 2 -c 5
#########################################################################
# 用于初始化每个工作线程的随机种子
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args):
    """
    训练循环
    :param args:
    """
    total_start_time = time.time()  # 记录总时间
    learning_rate = args.lr
    data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = args.classes
    save_checkpoint = args.save_checkpoint
    dir_checkpoint = args.checkpoint_path
    patience = args.patience
    mode = args.mode
    dir_checkpoint = os.path.join(dir_checkpoint, mode + "2D")

    # 读取配置文件
    config = configparser.ConfigParser()
    print("Loading config.ini")
    config.read('config2D.ini')  # 替换为配置文件路径
    train_dirs = config.get(args.mode, 'train_dir')
    train_dir = train_dirs.split(", ")
    val_dirs = config.get(args.mode, 'val_dir')
    val_dir = val_dirs.split(", ")

    label_intensities_str = config.get(args.mode, 'label_intensities')
    label_intensities = tuple(map(float, label_intensities_str.split(',')))
    class_to_pixel_str = config.get(args.mode, 'class_to_pixel')
    class_to_pixel = json.loads(class_to_pixel_str)

    ##########################
    #      dataloader        #
    ##########################
    try:
        start_time = time.time()
        print('start')
        ##########################
        #      train dataset     #
        ##########################
        print('Loading... ', train_dir[0])
        num_data = 5000
        source_dataset = Getfile(base_dir=data_path,
                                 data_dir=train_dir[0], num_classes=num_classes,
                                 label_intensities=label_intensities,
                                 mode=mode, onehot=True, num_data=num_data, aug=True)
        ##########################
        #      val dataset       #
        ##########################
        print('Loading... ', val_dir[0])
        source_val_dataset = Getfile(base_dir=data_path,
                                     data_dir=val_dir[0], num_classes=num_classes,
                                     label_intensities=label_intensities,
                                     mode=mode, onehot=False, num_data=0, aug=False)
        ##########################
        #      dataset       #
        ##########################
        train_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
        val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, drop_last=True)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"load time: {execution_time}s")

    except (AssertionError, RuntimeError, IndexError):
        raise RuntimeError("Failed to load the dataset.")

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    model.train()
    # 创建优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    Multi_loss = DiceCELoss(lambda_dice=0.5, lambda_ce=0.5)
    # 优化器、学习率调度器和损失已初始化，请完善训练循环
    ####################################################################################################################

    max_epoch = 0
    no_improve_count = 0
    best_val_score = float('-inf')
    for epoch in range(1, epochs + 1):
        save_flag = False
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit='batch')
        for batch in progress_bar:
            torch.cuda.empty_cache()
            volume_batch, label_batch = batch['image'].cuda(), batch['label'].cuda()
            masks_pred = model(volume_batch)
            loss = Multi_loss(masks_pred, label_batch.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            progress_bar.set_postfix(loss=loss)
            epoch_loss += loss
            # 将训练损失写入TensorBoard
            global_step = (epoch - 1) * len(train_loader) + progress_bar.n
            train_writer.add_scalar('Train/Loss', loss, global_step)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        print(f"Epoch [{epoch}/{epochs}], Current Learning Rate: {current_lr}")
        epoch_loss /= len(train_loader)
        logging.info('Epoch %d, average loss: %f', epoch, epoch_loss)
        val_score = evaluate(val_loader, model, num_classes, label_intensities, class_to_pixel, mode)
        val_writer.add_scalar('Validation/Score', val_score, epoch)
        logging.info('val_score: %f, epoch: %d', val_score, epoch)
        # 保存检查点
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        state = model.state_dict()
        if val_score > best_val_score:
            save_flag = True
            best_val_score = val_score
            max_epoch = epoch
            logging.info(f'Max_score is {best_val_score} at epoch{epoch}!')
            check_path = os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth')
            no_improve_count = 0  # 重置连续没有改进的周期计数器
        else:
            check_path = os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth')
            no_improve_count += 1
        logging.info('best_val_score: %f, epoch: %d', best_val_score, max_epoch)
        if save_flag or epoch % 20 == 0 or no_improve_count >= 30:
            torch.save(state, check_path)
        logging.info(f'{args.mode} Checkpoint {epoch} Saved!')
        # 如果连续没有改进的周期数超过了 patience，提前结束训练
        if no_improve_count > patience:
            logging.info(f'Early stopping triggered after {patience} epochs without improvement.')
            break
    ################################################################################################################
    # 记录总结束时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    formatted_total_time = str(timedelta(seconds=total_time))
    print(f'label_intensities : {label_intensities}')
    logging.info(f'Total Training Time: {formatted_total_time}')

    # 关闭两个SummaryWriter
    val_writer.close()
    train_writer.close()
    return "Training Finished!"


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    # 在train函数中设置两个监控，日志输出到指定位置
    train_writer = SummaryWriter(log_dir='logs/train')
    val_writer = SummaryWriter(log_dir='logs/val')
    # 初始化日志记录器
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{args.mode}.txt')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    # 固定随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 显卡相关设置
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    device_str = f'cuda:{gpu_device}' if gpu_device is not None else 'cpu'
    logging.info(f'Using device {device_str}')
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = UNet2D(in_channels=3, out_channels=args.classes).to(device)

    # 中断重新训练的加载功能
    if args.load != 0:
        checkpoint_path = os.path.join(args.checkpoint_path, 'checkpoint_epoch{}.pth'.format(args.load))
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from epoch {args.load}')
    model.to(device=device)
    train(args)
