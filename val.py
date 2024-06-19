import logging
import torch
import math
import numpy as np
import torch.nn.functional as F


def evaluate(val_dataset, net, num_classes, label_intensities, class_to_pixel, train_mode):
    net.eval()
    logging.info("Evaluating begin")
    # 仅记录非背景类别的分数
    sum_dice_scores = np.zeros(num_classes - 1)
    non_zero_counts = np.zeros(num_classes - 1)

    for idx, batch in enumerate(val_dataset):
        images, labels = batch['image'].cuda(), batch['label'].cuda()
        metrics = val_single_volume(images, labels, net, num_classes, class_to_pixel)
        for c, metric in enumerate(metrics):
            sum_dice_scores[c] += metric
            non_zero_counts[c] += 1

    avg_scores_except_first = np.mean(sum_dice_scores / non_zero_counts)
    avg_dice_scores_per_class = sum_dice_scores / non_zero_counts
    # 按类别输出验证结果
    for c in range(0, num_classes - 1):
        logging.info(f"Avg Dice Score for Class {c + 1}: {float(avg_dice_scores_per_class[c])}")
    logging.info(f"Avg Dice Score for Classes (excluding Class 0): {float(avg_scores_except_first)}")
    logging.info("Evaluating end")

    return avg_scores_except_first


def val_single_volume(images, labels, net, num_classes, class_to_pixel):
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

    metric_list = [calculate_metric_percase(predictions == i, labels == i) for i in range(1, num_classes)]
    return metric_list


def calculate_metric_percase(pred, gt):
    pred = pred.float()
    gt = gt.float()
    intersection = (pred * gt).sum()
    dice = (2.0 * intersection) / (pred.sum() + gt.sum())
    return dice.item()
