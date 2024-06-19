import os
import glob
import sys
import random
import monai
import numpy as np
from matplotlib import pyplot as plt
from monai.transforms import LoadImage, Orientation, Zoom
from tqdm import tqdm

from utils.preprocess_bst import winadj_mri

sys.path.append("..")
# from utils import read_list, read_nifti, config
import SimpleITK as sitk
from scipy.ndimage import zoom


def write_txt(data, path):
    with open(path, 'w') as f:
        for val in data:
            f.writelines(val + '\n')


def convert_labels(label):
    label[label == 205] = 1
    label[label == 420] = 2
    label[label == 500] = 3
    label[label == 820] = 4
    label[label > 4] = 0
    return label


def adjust_contrast(image_arr, contrast_factor=1.0):
    """
    Returns:
    numpy.ndarray: The adjusted image array.
    """
    # Calculate the mean pixel value
    mean_val = image_arr.mean()

    # Apply the contrast adjustment
    adjusted_image = (image_arr - mean_val) * contrast_factor + mean_val

    # Clip values to valid range
    # adjusted_image = np.clip(adjusted_image, 0, 1)

    return adjusted_image


def getRangeImageDepth(label):
    d = np.any(label, axis=(1, 2))
    h = np.any(label, axis=(0, 2))
    w = np.any(label, axis=(0, 1))

    if len(np.where(d)[0]) > 0:
        d_s, d_e = np.where(d)[0][[0, -1]]
    else:
        d_s = d_e = 0

    if len(np.where(h)[0]) > 0:
        h_s, h_e = np.where(h)[0][[0, -1]]
    else:
        h_s = h_e = 0

    if len(np.where(w)[0]) > 0:
        w_s, w_e = np.where(w)[0][[0, -1]]
    else:
        w_s = w_e = 0
    return d_s + 5, d_e - 5, h_s + 5, h_e - 5, w_s + 5, w_e - 5


def process_npy():
    mode = 'ct_train'
    base_dir = 'D:/DeskTop/data/MM-WHS/' + mode
    save_dir = 'D:/DeskTop/data/MMWHS2D/' + mode
    minpix = 99
    maxpix = 0
    for f in os.listdir(base_dir):
        if f.endswith('image.nii.gz'):
            image_path = os.path.join(base_dir, f)
            labelf = f.replace('image', 'label')
            label_path = os.path.join(base_dir, labelf)
            # print(f)

            # if not os.path.exists(os.path.join(save_dir, 'processed')):
            #     os.makedirs(os.path.join(save_dir, 'processed'))

            # 使用 MONAI 的 LoadImage 加载图像
            image = LoadImage()(image_path).unsqueeze(0)
            label = LoadImage()(label_path).unsqueeze(0)

            # 使用 Orientation 调整方向到 RAI
            image = Orientation(axcodes='SPR')(image)
            label = Orientation(axcodes='SPR')(label)

            # 转换为 NumPy 数组
            image_arr = image.squeeze(0).numpy()
            label_arr = label.squeeze(0).numpy()
            label_arr = convert_labels(label_arr)
            if f == "mr_train_1002_image.nii.gz":
                # label_arr[0:10, :, :] = 0
                label_arr[:, -50:-1, :] = 0
                # label_arr[:, :, 0:10] = 0
                # label_arr[-10:-1, :, :] = 0
                label_arr[:, 0:50, :] = 0
                # label_arr[:, :, -10:-1] = 0
            if f == "ct_train_1004_image.nii.gz":
                label_arr[0:10, :, :] = 0
                # label_arr[:, -10:-1, :] = 0
                label_arr[:, :, 0:10] = 0
                label_arr[-10:-1, :, :] = 0
                # label_arr[:, 0:10, :] = 0
                label_arr[:, :, -10:-1] = 0

            image_arr = image_arr.astype(np.float32)
            # 获取标签的有效范围
            d_s, d_e, h_s, h_e, w_s, w_e = getRangeImageDepth(label_arr)
            # 裁剪图像和标签
            image_arr = image_arr[d_s:d_e, h_s:h_e, w_s: w_e]
            label_arr = label_arr[d_s:d_e, h_s:h_e, w_s: w_e]

            # 使用 Zoom 调整大小到 256x256x256
            dn, hn, wn = image_arr.shape
            image_arr = zoom(image_arr, [256 / dn, 256 / hn, 256 / wn], order=0)
            label_arr = zoom(label_arr, [256 / dn, 256 / hn, 256 / wn], order=0)

            # 归一化图像
            # upper_bound_intensity_level = np.percentile(image_arr, 98)
            # min_percentile = np.percentile(image_arr, 1)
            # max_percentile = np.percentile(image_arr, 99)
            # image_arr = np.clip(image_arr, min_percentile, max_percentile)
            # image_arr = (image_arr - min_percentile) / (max_percentile - min_percentile) * 2 - 1
            # if img_id == 'ct_train_32_image':
            #     upper_bound_intensity_level = np.percentile(image_arr, 95)
            #     print(32)
            # if img_id == 'ct_train_26_image':
            #     upper_bound_intensity_level = np.percentile(image_arr, 98.5)
            #     print(26)
            # if img_id == 'mr_train_36_image':
            #     upper_bound_intensity_level = np.percentile(image_arr, 99)
            #     print(26)
            # image_arr = image_arr.clip(min=0, max=upper_bound_intensity_level)
            # image_arr = (image_arr - image_arr.mean()) / (image_arr.std() + 1e-8)
            # 参考FPL+,获取3D图像的最大最小值做归一化
            image_arr = winadj_mri(image_arr)
            # image_slice = 2 * ((image_slice - mode_min) / (mode_max - mode_min)) - 1
            #
            # # 计算原始数据范围的最大值和最小值
            # min_value = image_arr.min()
            # max_value = image_arr.max()
            # minpix = min(minpix, min_value)
            # maxpix = max(maxpix, max_value)
            # 生成原始切片
            imgname = f.replace('.nii.gz', '')
            labelname = labelf.replace('.nii.gz', '')
            txt_file_path = os.path.join(save_dir, f'{mode}.txt')
            with open(txt_file_path, 'a') as txt_file:  # 使用追加模式
                for z in range(0, 254, 1):
                    image_slice = image_arr[:, z:z + 3, :].transpose((1, 0, 2))
                    label_slice = label_arr[:, z + 1, :].reshape(256, 1, 256).transpose((1, 0, 2))  # 使用中间切片的掩模标签

                    ####################################################
                    # print(image_slice.shape, label_slice.shape)
                    """
                    ct_train:[7, 13, 1, 4, 15, 8, 14, 18, 3, 12, 2, 16, 20, 5]
                    -1.046097 2.5861375
                    
                    ct_val:[6, 10]
                    -0.94972354 2.1397567
                    
                    ct_test:[17, 9, 11, 19]
                    -0.8716146 2.8582082
                    
                    mr_train:[7, 13, 1, 4, 15, 8, 14, 18, 3, 12, 2, 16, 20, 5]
                    -1.6842501 2.277872
                    
                    mr_val:[6, 10]
                    -1.4655517 1.8819735
                    
                    mr_test:[17, 9, 11, 19]
                    -1.5968806 2.2330086
                    """
                    # mode_min, mode_max = -1.1, 2.9  # ct
                    # mode_min, mode_max = -1.7, 2.3  # mr
                    ####################################################

                    
                    merged_image = np.hstack((image_slice[1, :, :], label_slice[0]))

                    # 将图像保存为灰度图像
                    save_path = os.path.join('D:/DeskTop/check', f'{imgname}_{z}.png')
                    plt.imsave(save_path, merged_image, cmap='gray')


                    #################################################################################
                    # 保存为.npy文件
                    # np.save(os.path.join(save_dir,f'{imgname}_{z}.npy' ), image_slice)
                    # np.save(os.path.join(save_dir, f'{labelname}_{z}.npy'), label_slice) # 保存图像和标签到.npz文件
                    #################################################################################

                    #
                    # txt_file.write(f'{imgname}_{z}.npz' + ',' + '\n')
                    # np.savez(os.path.join(save_dir, f'{imgname}_{z}.npz'), image=image_slice, label=label_slice)


                    ##########################################
                    # print(image_slice.shape, label_slice.shape)
                    # min_value = image_slice.min()
                    # max_value = image_slice.max()
                    # minpix = min(minpix, min_value)
                    # maxpix = max(maxpix, max_value)
                    # if z % 100 == 0:
                    #     plt.imshow(image_slice[:, 1, :], cmap='gray')  # 显示图像
                    #     plt.show()
                    #     plt.imshow(label_slice[:, 0, :], cmap='gray')
                    #     plt.show()
                # print(minpix, maxpix)
###############################################33

if __name__ == '__main__':
    # # 设置随机种子
    # random.seed(8888)
    #
    # # 生成1到20的列表
    # numbers = list(range(1, 21))
    #
    # # 随机打乱列表
    # random.shuffle(numbers)
    #
    # # 计算每个部分的大小
    # total_size = len(numbers)
    # part_sizes = [14, 2, 4]
    #
    # # 划分数据
    # data_splits = []
    # start = 0
    # for size in part_sizes:
    #     data_splits.append(numbers[start:start + size])
    #     start += size
    #
    # # 打印结果
    # for i, split in enumerate(data_splits):
    #     print(f"Part {i + 1}: {split}")

    process_npy()
    # process_split_fully()

    # npz_file_path = 'D:/DeskTop/data/MMWHS2D/mr_val/mr_train_1006_image_200.npz'
    #
    # # 使用np.load读取npz文件
    # with np.load(npz_file_path) as data:
    #     image_slice = data['image']
    #     label_slice = data['label']
    #     plt.imshow(image_slice[1, :, :], cmap='gray')  # 显示图像
    #     plt.show()
    #     plt.imshow(label_slice[0, :, :], cmap='gray')
    #     plt.show()
