import argparse
# import torch.optim as optim
from PIL import Image
import torch.utils.data
from torch.utils.data import DataLoader
import  numpy as np
from os.path import join
from tqdm import tqdm
# from torchvision.transforms import RandomCrop, ToTensor, ToPILImage
#
# from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale
# # import pytorch_ssim
from old_codes.data_utils import TestDatasetFromFolder
from old_codes.model_8 import Generator
# from data_utils import display_transform
# from model_fnet import FusionNet

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=1, type=int, help='train epoch number')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size  # 裁剪会带来拼尽问题嘛
UPSCALE_FACTOR = opt.upscale_factor  # 上采样
NUM_EPOCHS = opt.num_epochs  # 轮数

sate = 'pl'
dataset_dir = 'E:\\remote sense image fusion\\Source Images\\'
# dataset_dir = r'E:\py_code\pycode_lu1_JX3\Dataset/'

val_set = TestDatasetFromFolder(dataset_dir, sate, upscale_factor=UPSCALE_FACTOR)  # 测试集导入  # change
# val_set = TestDatasetFromFolder('E:\\py_code\\pycode_lu1_JX3\\Dataset/', upscale_factor=UPSCALE_FACTOR)  # 测试集导入  # change

MODEL_NAME = 'netG_epoch_1_10.pth'
netG = Generator(UPSCALE_FACTOR).eval()
# netG = FusionNet()
# netG.cuda()      # change
# netG.load_state_dict(torch.load('D:/Project/NewProject/ResNetFusion-master/model_trained/' + MODEL_NAME))

netG.load_state_dict(torch.load('E:\\py_code\\pycode_lu1_JX3/model_trained/' + MODEL_NAME, map_location='cpu'))  # changed
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

epoch = 1
out_path = 'E:\\py_code\\pycode_lu1_JX3\\Dataset/results_' + str(UPSCALE_FACTOR) + '/'  # 输出路径

val_bar = tqdm(val_loader)  # 验证集的进度条
val_images = []  # 表示一个列表


# def gain(image):
#     c, w, h = image.shape
#     sum_image = torch.zeros(1, w, h)
#     for i in range(c):
#         sum_image += image[i]
#     sum_image = 1 / c * torch.cat([sum_image, sum_image, sum_image, sum_image], 0)
#     g = image / sum_image
#     return g

def gain(image):
    n, c, w, h = image.shape
    # sum_image[1, w, h] = torch.zeros(1, w, h)
    g = torch.zeros(n, c, w, h)
    for i in range(n):
        sum_image = torch.zeros(1, 1, w, h)
        for j in range(c):
            sum_image += image[i, j]
        sum_image = 1 / c * torch.cat([sum_image, sum_image, sum_image, sum_image], 1)
        sum_image = sum_image.squeeze()
        g[i] = image[i] / sum_image
    return g


num = 100
for data_cat, ms_gray, pan, ms_up, gt, gt_gray in val_bar:

    g = gain(ms_up)
    num += 1
    data = data_cat
    # output = netG(pan.float(), ms_gray.float())
    output = netG(data.float())


    # output = output.cpu()
    for i in range(output.data.shape[0]):
        # fused = output + ms_up
        # fused = ms_up + 2 * g * output * (pan - ms_gray)
        # fused = (fused.data[i] + 1) / 2.0
        # fused = ms_up + 1.5 * g * (output-ms_up)
        # ms_gray = ms_gray.mul(255).byte()
        # ms_gray = ms_gray.squeeze(0)
        # ms_gray = np.transpose(ms_gray.numpy(), (1, 2, 0))
        # ms_gray = ms_gray.squeeze(2)

        fused = output
        fused = fused.mul(255).byte()
        fused = fused.squeeze(0)
        fused = np.transpose(fused.detach().numpy(), (1, 2, 0))
        fused = fused.squeeze(2)
        # fused = (fused - np.mean(fused)) * np.std(ms_gray) / np.std(fused) + np.mean(ms_gray)       #
        # fused = (fused * 255).astype(int)

        gt_gray = gt_gray.mul(255).byte()
        gt_gray = gt_gray.squeeze(0)
        gt_gray = np.transpose(gt_gray.numpy(), (1, 2, 0))
        gt_gray = gt_gray.squeeze(2)

        pan = pan.mul(255).byte()
        pan = pan.squeeze(0)
        pan = np.transpose(pan.numpy(), (1, 2, 0))
        pan = pan.squeeze(2)


        save_f_name = '%d_out_tf.tif' % num
        save_gt_name = '%d_gt_tf.tif' % num
        save_pan_name = '%d_pan_tf.tif' % num

        # save_image(sate, save_f_name, fused)

        save_path = join(r'E:\py_code\pycode_lu1_JX3\saved images', save_f_name)
        fused_pil = Image.fromarray(fused)
        fused_pil.save(save_path)

        save_path2 = join(r'E:\py_code\pycode_lu1_JX3\saved images', save_gt_name)
        gt_pil = Image.fromarray(gt_gray)
        gt_pil.save(save_path2)

        save_path3 = join(r'E:\py_code\pycode_lu1_JX3\saved images', save_pan_name)
        pan_pil = Image.fromarray(pan)
        pan_pil.save(save_path3)




    # batch_size = gray_cat.size(0)
    # gray_input = Variable(gray_cat)  # concatenate of PAN　and intensity
    # hr = Variable(pan)  # PAN
    # if torch.cuda.is_available():
    #     gray_input = gray_input.cuda()
    #     hr = hr.cuda()
    #
    # sr = netG(gray_input)  # 验证集生成超分图片  # 3 * 256 *256
    #
    # val_images.extend(
    #     [display_transform()(ms_gray.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
    #      display_transform()(sr.data.cpu().squeeze(0))])  # 列表扩展 list 3 *1 * 256 *256
    #
    # ms_up = ms_up.squeeze(0)
    # ms_gray = val_images[0 + num * 3]
    # pan = val_images[1 + num * 3]  # 1*256*256
    #
    # ms_gray4 = torch.cat([ms_gray, ms_gray, ms_gray, ms_gray], 0)
    # pan4 = torch.cat([pan, pan, pan, pan])  # 4*256*256
    # weight = val_images[2 + num * 3]
    # weight = torch.cat([weight, weight, weight, weight], 0)
    #
    # # unloader = transforms.ToPILImage()
    # # pan1 = pan.squeeze(0)     # 256*256
    # # pan1 = unloader(pan1)
    # # pan1.show()
    # #
    # # ms_gray1 = ms_gray.squeeze(0)  # 256*256
    # # ms_gray1 = unloader(ms_gray1)
    # # ms_gray1.show()
    #
    # g = gain(ms_up)
    # # fused = ms_up + g * 1.8 * weight*(pan4 - ms_gray4)
    # # fused = ms_up + 0.8 * g * (pan4 - ms_gray4)
    # # fused = ms_up + (pan4 - ms_gray4)
    #
    # fused = fused.transpose(0, 2).transpose(0, 1).contiguous()
    # num += 1
    # file_name = '%d.tif' % (100 + num)
    # save_image(sate, file_name, fused)

# val_images = torch.stack(val_images)
# # val_images = torch.chunk(val_images, val_images.size(0) // 15)#看不懂，骚操作
# val_images = torch.split(val_images, 1, dim=0)  # 变成元组


# start = time.time()
# for i, image in enumerate(val_images):  # list is ok
#     print('{}th size {}'.format(i, image.size()))
# val_save_bar = tqdm(val_images, desc='[saving training results]')
# index = 1
# for image in val_save_bar:
#     # image = utils.make_grid(image, nrow=3, padding=2, scale_each=True)
#     # utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), nrow=3, padding=2)  # 验证集存储数据
#     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index))  # 验证集存储数据
#     end = time.time()
#     print(end - start)
#     index += 1
