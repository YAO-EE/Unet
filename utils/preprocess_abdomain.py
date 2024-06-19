import os
import numpy as np
import matplotlib.pyplot as plt


def _decode_samples(npz_path, file_indices, mode, save_dir):
    def load_npz(file_path):
        data = np.load(file_path)
        # 获取数据
        data_vol = data['arr_0.npy']
        label_vol = data['arr_1.npy']
        return data_vol, label_vol

    dataset = []
    txt_file_path = os.path.join(save_dir, f'{mode}.txt')
    minpix = 99
    maxpix = 0
    with open(txt_file_path, 'a') as txt_file:  # 使用追加模式
        for idx in file_indices:
            folder_path = os.path.join(npz_path, str(idx))
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.npz'):
                        file_path = os.path.join(folder_path, file_name)
                        data_vol, label_vol = load_npz(file_path)
                        """
                        ct
                        -1.1844577 3.4539323
                        -1.2810926 2.7264569
                        -1.1304629 3.0022898
                        mr
                        -1.1623437 4.1146436
                        -1.1547594 3.8114905
                        -1.1277238 4.1356735
                        """
                        # mode_min, mode_max = -1.3, 3.5  # ct
                        mode_min, mode_max = -1.2, 4.2  # mr
                        data_vol = 2 * ((data_vol - mode_min) / (mode_max - mode_min)) - 1
                        # plt.imshow(data_vol, cmap='gray')  # 显示图像
                        # plt.show()
                        # plt.imshow(label_vol, cmap='gray')
                        # plt.show()
                        # min_value = data_vol.min()
                        # max_value = data_vol.max()
                        # minpix = min(minpix, min_value)
                        # maxpix = max(maxpix, max_value)
                        # 复制data_vol和label_vol
                        data_vol_copied = data_vol[np.newaxis, :, :].repeat(3, axis=0)
                        label_vol_copied = label_vol[np.newaxis, :, :]
                        # 保存修改后的数据
                        txt_file.write(f'mr_train_{idx}_image_{file_name}' + ',' + '\n')
                        np.savez(os.path.join(save_dir, f'mr_train_{idx}_image_{file_name}'), image=data_vol_copied,
                                 label=label_vol_copied)
                        dataset.append((data_vol, label_vol))
        # print(minpix, maxpix)
    # print(len(dataset))
    return dataset


def save_as_npy(data, save_dir):
    prefix = 'mr_test'
    count = 0
    min_gray = float('inf')  # 初始设置最小灰度值为正无穷大
    max_gray = float('-inf')  # 初始设置最大灰度值为负无穷大

    with open(os.path.join(save_dir, 'mr_test_list.txt'), 'a') as file_list:  # 使用追加模式
        for i, (image, label) in enumerate(data):
            # 保存数据
            np.save(os.path.join(save_dir, f'{prefix}_image_{count}.npy'), image)
            np.save(os.path.join(save_dir, f'{prefix}_label_{count}.npy'), label)
            file_list.write(f'{prefix}_image_{count}.npy\n')

            # 更新最小和最大灰度值
            min_gray = min(min_gray, np.min(image))
            max_gray = max(max_gray, np.max(image))

            count += 1
            if count % 100 == 0:
                plt.imshow(image)  # 显示图像
                plt.show()
                plt.imshow(label)
                plt.show()

    print("Min Gray:", min_gray)
    print("Max Gray:", max_gray)


if __name__ == '__main__':
    mode = 'mr_test'
    save_dir = 'D:/DeskTop/data/abdomin2D/' + mode
    npz_path = 'D:/DeskTop/data/abdomin/abdominalDATA/MR_T2_npy'  # npz文件夹的路径
    # save_dir = 'D:/DeskTop/data/abdomin/abdominalDATA/mr_test'
    # file_indices = ['02', '04', '05', '06', '07', '08', '10', '21', '22', '23', '25', '26', '27', '28', '29', '31', '34', '35', '37', '38', '40']  # CT_train
    # file_indices = ['03', '24', '36']  # ct_val
    # file_indices = ['01', '09', '30', '32', '33', '39']  # CT_test
    # file_indices = ['2', '3',  '8', '10', '15', '19', '20', '21', '31', '33', '34', '36', '37', '39']  # MR_train
    # file_indices = ['5', '22']  # MR_val
    file_indices = ['1', '13', '32', '38']  # MR_test

    data = _decode_samples(npz_path, file_indices, mode, save_dir)
    # save_as_npy(data, save_dir)
