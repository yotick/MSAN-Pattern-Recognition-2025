import torch
import matplotlib.pyplot as plt
import os

# 加载保存在.pt文件中的张量
# file_names = ['good_35_conv15.pt','good_35_swin.pt','good_35_comb.pt','good_35_res.pt']
# file_names = ['good_35_conv30.pt','good_35_swin.pt','good_35_comb1.pt']
file_names = ['good_35_swin.pt']
# file_name = 'good_35_res.pt'     #### need to change here ####################

# tensor = torch.load('output_0_swin.pt').cpu()

for i in range(2):
    # 选择第一个通道的图像
    channel = 39+i*5            #### need to change here ####################
    for file_name in file_names:
        tensor = torch.load('visualize\\' + file_name).cpu()
        image = tensor[0, channel, :, :]

        # 将张量转换为NumPy数组，并将像素值缩放到[0,1]之间
        image = image.numpy()
        image = (image - image.min()) /(image.max() - image.min())

        # 指定颜色映射范围
        vmin, vmax = image.min(), image.max()

        # 设置颜色映射
        cmap = plt.cm.get_cmap('jet')

        # 使用Matplotlib库将图像可视化
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        # plt.colorbar()
        # 保存热力图
        # 使用os.path.splitext函数获取文件名的左边部分
        file_name_left = os.path.splitext(file_name)[0]
        save_name = 'htmap_' + '_ch{}_'.format(channel) + file_name_left
        plt.savefig('visualize\\' + save_name, bbox_inches='tight')

plt.show()
