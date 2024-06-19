import csv
import os
import random
from posixpath import split
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from monai.transforms import LoadImage, Orientation
from scipy.ndimage import zoom

from utils.preprocess_mmwhs import getRangeImageDepth


def winadj_mri(array):
    v0 = np.percentile(array, 1)
    v1 = np.percentile(array, 999)
    array[array < v0] = v0
    array[array > v1] = v1
    v0 = array.min()
    v1 = array.max()
    array = (array - v0) / (v1 - v0) * 2.0 - 1.0
    return array


def random_partition_numbers(start, end, num):
    """
    将[start, end]之间的整数随机划分成两部分，并且每个数字转换为三位数的字符串形式
    :param start: 开始数字
    :param end: 结束数字
    :param num: 第一部分数字的数量
    :return: 两部分数字列表
    143/21/ 41-flair
    143/21/ 41-t2
    """
    random.seed(8888)
    # 生成数字列表，并将每个数字转换为三位数的字符串形式
    numbers = [f"{i:03d}" for i in range(start, end + 1)]

    # 随机打乱数字列表
    random.shuffle(numbers)

    # 划分数字列表
    first_partition = numbers[:num]
    second_partition = numbers[num:]

    return first_partition, second_partition


def crop_depth(img, lab):
    D, W, H = img.shape
    nonzero_indices = np.nonzero(lab)
    d_min, d_max = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    d_margin = 0
    # 确定要去除的边界
    if d_max - d_min > 35:
        d_margin = 16  # 在深度方向上的边界宽度
    d00 = max(d_min + d_margin, 0)
    d11 = min(d_max - d_margin, D)
    zero_img = img[d00:d11, :, :]
    zero_lab = lab[d00:d11, :, :]
    return zero_img, zero_lab


def resize_and_crop(image_arr, label_arr, target_size=240):
    # 计算非黑边区域的边界框
    nonzero_indices = np.where(image_arr != 0)
    min_row, max_row = np.min(nonzero_indices[-2]), np.max(nonzero_indices[-2])
    min_col, max_col = np.min(nonzero_indices[-1]), np.max(nonzero_indices[-1])

    # 计算宽度和高度的最大值，并计算放大比例
    max_width = max_row - min_row
    max_height = max_col - min_col
    max_dim = max(max_width, max_height)
    zoom_factor = target_size / (max_dim + 20)

    # 在后两个维度上等比例放大
    image_arr = zoom(image_arr, zoom=[1, zoom_factor, zoom_factor], order=0)
    label_arr = zoom(label_arr, zoom=[1, zoom_factor, zoom_factor], order=0)

    # 获取放大后的图像形状
    resized_shape = image_arr.shape

    crop_x = int((resized_shape[1] - target_size) / 2)
    crop_y = int((resized_shape[2] - target_size) / 2)

    # 裁剪图像和标签
    cropped_image_arr = image_arr[:, crop_x + 10:crop_x + target_size + 10, crop_y:crop_y + target_size]
    cropped_label_arr = label_arr[:, crop_x + 10:crop_x + target_size + 10, crop_y:crop_y + target_size]

    return cropped_image_arr, cropped_label_arr


# def remove_black_borders(img, lab):
#     # 计算在深度方向上非零像素的范围
#     D, W, H = img.shape
#     nonzero_indices = np.nonzero(lab)
#     d_min, d_max = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
#
#     # 计算在宽度和高度方向上非零像素的范围
#     w_indices = np.any(img, axis=(0, 2))
#     h_indices = np.any(img, axis=(0, 1))
#     w_min, w_max = np.where(w_indices)[0][[0, -1]]
#     h_min, h_max = np.where(h_indices)[0][[0, -1]]
#
#     d_margin = 0
#     # 确定要去除的边界
#     if d_max - d_min > 35:
#         d_margin = 16  # 在深度方向上的边界宽度
#     w_margin = 5  # 在宽度方向上的边界宽度
#     h_margin = 5  # 在高度方向上的边界宽度
#
#     # 计算裁剪的位置
#     d00 = max(d_min + d_margin, 0)
#     d11 = min(d_max - d_margin, D)
#     w00 = max(w_min - w_margin, 0)
#     w11 = min(w_max + w_margin, W)
#     h00 = max(h_min - h_margin, 0)
#     h11 = min(h_max + h_margin, H)
#
#     # 执行裁剪
#     cropped_img = img[d00:d11, w00:w11, h00:h11]
#     cropped_lab = lab[d00:d11, w00:w11, h00:h11]
#
#     return cropped_img, cropped_lab


# def crop_depth(img, lab):
#     D, W, H = img.shape
#     nonzero_indices = np.nonzero(lab)  # 获取标签非零元素的索引
#     d_min, d_max = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])  # 获取标签在深度方向上的范围
#     margin = 5
#     if (d_max - d_min) > 35:
#         margin = int((d_max - d_min) * 0.2)  # 计算裁剪的边界宽度，取标签深度范围的百分之10
#     d00 = max(d_min + margin, 0)  # 裁剪的起始深度位置
#     d11 = min(d_max - margin, D)  # 裁剪的结束深度位置
#     zero_img = img[d00:d11, :, :]  # 根据计算得到的位置进行裁剪
#     zero_lab = lab[d00:d11, :, :]
#     return zero_img, zero_lab


def save_files(start, length, model, mode):
    minpix = 99
    maxpix = 0
    flag = False
    while length > 0:
        num = numbers[start]  # 对应数据文件夹的标号
        roots = 'D:/DeskTop/data/MICCAI_BraTS2020_TrainingData'  # 源数据根目录
        commen = 'BraTS20_Training'  # 共有文件名
        base_dir = 'D:/DeskTop/data/BraTs2D'
        save_dir = os.path.join(base_dir, model)

        image_path = roots + '/' + commen + '_' + num + '/' + commen + '_' + num + '_' + mode + '.nii'
        label_path = roots + '/' + commen + '_' + num + '/' + commen + '_' + num + '_seg' + '.nii'

        image = LoadImage()(image_path).unsqueeze(0)
        label = LoadImage()(label_path).unsqueeze(0)
        image = Orientation(axcodes='SPR')(image)
        label = Orientation(axcodes='SPR')(label)
        # 转换为 NumPy 数组
        image_arr = image.squeeze(0).numpy()
        label_arr = label.squeeze(0).numpy()

        label_arr[label_arr > 0] = 1

        image_arr = image_arr.astype(np.float32)
        image_arr, label_arr = crop_depth(image_arr, label_arr)
        # print(num, image_arr.shape)
        image_arr, label_arr = resize_and_crop(image_arr, label_arr, target_size=240)

        # print(num, image_arr.shape)
        upper_bound_intensity_level = np.percentile(image_arr, 98)
        image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
        image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)
        """
        flair:
        -1.0023208 3.5509555
        t2:
        -0.99422425 4.0128164

        """
        if mode == 'flair':
            mode_min, mode_max = -1.1, 3.6  # flair
        else:
            mode_min, mode_max = -1.0, 4.1  # t2
        txt_file_path = os.path.join(save_dir, f'{model}.txt')
        with open(txt_file_path, 'a') as txt_file:  # 使用追加模式
            for z in range(0, image_arr.shape[0] - 2, 1):
                image_slice = image_arr[z:z + 3, :, :]
                label_slice = label_arr[z + 1, :, :]  # 使用中间切片的掩模标签

                label_slice = label_slice.reshape(1, 240, 240)
                if np.sum(label_slice) == 0:
                    continue

                image_slice = 2 * ((image_slice - mode_min) / (mode_max - mode_min)) - 1
                min_value = image_slice.min()
                max_value = image_slice.max()
                minpix = min(minpix, min_value)
                maxpix = max(maxpix, max_value)

                txt_file.write(f'{model}_{num}_image_{z}.npz' + ',' + '\n')
                np.savez(os.path.join(save_dir, f'{model}_{num}_image_{z}'), image=image_slice,
                         label=label_slice)
                if flag:
                    break

        start += 1
        length -= 1
        if model == 't_test' or model == 'f_test':
            if length == 1:
                start = 0
                length = 1
                flag = True
    print(minpix, maxpix)


if __name__ == "__main__":
    img_path = 'D:/DeskTop/data/MICCAI_BraTS2020_TrainingData'
    # save_img = '/your/bst_data/' + moda + '_' + phase + '/img'
    # save_lab = '/your/bst_data/' + moda + '_' + phase + '/lab'
    # if (not os.path.exists(save_lab)):
    #     os.mkdir(save_lab)
    names = os.listdir(img_path)

    random.seed(8888)  # 设置随机数种子（可选，用于重现结果）
    # 生成数字列表，并将每个数字转换为三位数的字符串形式
    numbers = [f"{i:03d}" for i in range(1, 369 + 1)]
    random.shuffle(numbers)

    """
    143/21/ 41-flair
    143/21/ 41-t2
    1-143
    144-286
    287-307
    308-328
    329-369
    329-369
    """
    start = 329
    length = 41
    model = 't_test'  # 保存的数据的文件夹
    # f_train  f_val  f_test  t_train  t_val  t_test
    mode = 't2'  # 加载数据的模态
    # flair  t_train
    save_files(start, length, model, mode)

    # plt.imshow(image_slice[1, :, :], cmap='gray')  # 显示图像
    # plt.show()
    # plt.imshow(label_slice[0, :, :], cmap='gray')
    # plt.show()
    # if z <= 30 or z >= 245:
    #     merged_image = np.hstack((image_slice[1, :, :], label_slice[0]))
    #
    #     # 将图像保存为灰度图像
    #     save_path = os.path.join('D:/DeskTop/check', f'{caase_name}+{z}.png')
    #     plt.imsave(save_path, merged_image, cmap='gray')

    # np.savez(os.path.join(save_dir, f'{mode}_train_{caase_name}_image_{idx}'), image=img_obj,
    #          label=lab_obj)
    # img_save_dir = os.path.join(save_img, caase_name + '.nii.gz')
    # lab_save_dir = os.path.join(save_lab, caase_name + '.nii.gz')
    # sitk.WriteImage(img_obj, img_save_dir)
    # sitk.WriteImage(lab_obj, lab_save_dir)
