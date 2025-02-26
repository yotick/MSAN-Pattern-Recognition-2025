import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_set_py.data_utils_Full import TestDatasetFromFolder_Full
# from data import get_test_set
# from model_8_1 import Generator  ################ need to change ######################
# from models.model_4_swin3 import Generator  ################ need to change ######################
from models.model_MSAN_8C import MSAN  ################ need to change ######################
# sys.path.append('E:\\remote sense image fusion\\shared_py')
# from data_set_py.save_image_func import save_image_full
import time
from save_image_func import save_image_full

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def overlap_crop_forward(ms_up, ms, pan, shave=16, min_size=68200, bic=None):
    """
    chop for less memory consumption during test
    """
    n_GPUs = 1
    scale = 1

    # y = ms
    # z = pan
    b, c, h, w = ms_up.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave  # 一半且加上叠加部分
    ms_up_list = [
        ms_up[:, :, 0:h_size, 0:w_size],  # 左上
        ms_up[:, :, 0:h_size, (w - w_size):w],  # 右上
        ms_up[:, :, (h - h_size):h, 0:w_size],  # 左下
        ms_up[:, :, (h - h_size):h, (w - w_size):w]]  # 右下

    h_size_org, w_size_org = h_size // 4, w_size // 4  # 一半且加上叠加部分
    ms_org_list = [
        ms[:, :, 0:h_size_org, 0:w_size_org],  # 左上
        ms[:, :, 0:h_size_org, (w // 4 - w_size_org):w // 4],  # 右上
        ms[:, :, (h // 4 - h_size_org):h // 4, 0:w_size_org],  # 左下
        ms[:, :, (h // 4 - h_size_org):h // 4, (w // 4 - w_size_org):w // 4]]  # 右下

    pan_list = [
        pan[:, :, 0:h_size, 0:w_size],  # 左上
        pan[:, :, 0:h_size, (w - w_size):w],  # 右上
        pan[:, :, (h - h_size):h, 0:w_size],  # 左下
        pan[:, :, (h - h_size):h, (w - w_size):w]]  # 右下

    if bic is not None:  # 缩放？可以忽略
        bic_h_size = h_size * scale
        bic_w_size = w_size * scale
        bic_h = h * scale
        bic_w = w * scale

        bic_list = [
            bic[:, :, 0:bic_h_size, 0:bic_w_size],
            bic[:, :, 0:bic_h_size, (bic_w - bic_w_size):bic_w],
            bic[:, :, (bic_h - bic_h_size):bic_h, 0:bic_w_size],
            bic[:, :, (bic_h - bic_h_size):bic_h, (bic_w - bic_w_size):bic_w]]

    if w_size * h_size < min_size:  # 如果已经小于最小尺度
        sr_list = []
        for i in range(0, 4, n_GPUs):  # 分成四部分
            ms_up_batch = torch.cat(ms_up_list[i:(i + n_GPUs)], dim=0)  # crop 之后作为 batch
            ms_batch = torch.cat(ms_org_list[i:(i + n_GPUs)], dim=0)  # crop 之后作为 batch
            pan_batch = torch.cat(pan_list[i:(i + n_GPUs)], dim=0)  # crop 之后作为 batch

            if bic is not None:
                bic_batch = torch.cat(bic_list[i:(i + n_GPUs)], dim=0)

            sr_batch_temp = model(ms_up_batch, ms_batch, pan_batch)  # only 1 batch

            if isinstance(sr_batch_temp, list):
                sr_batch = sr_batch_temp[-1]
            else:
                sr_batch = sr_batch_temp

            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))  ##　拆分
    else:
        sr_list = [
            overlap_crop_forward(ms_up_p, ms_p, pan_p, shave=shave, min_size=min_size) \
            for ms_up_p, ms_p, pan_p in zip(ms_up_list, ms_org_list, pan_list)
            # for ms_up_p in ms_up_list  # 递归调用切割
            # for ms_p in ms_org_list
            # for pan_p in pan_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = ms_up.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def test(test_data_loader, model, sate):
    val_bar = tqdm(test_data_loader)  # 验证集的进度条
    count = 0
    for ms_up_crop, ms_org_crop, pan_crop, _, _ in val_bar:
        batch_size = ms_up_crop.size(0)  # current batch size
        # valing_results['batch_sizes'] += batch_size
        # detail_crop = detail_crop.type(torch.FloatTensor)  # to make the type the same as model
        # data = torch.cat(ms_up_crop, pan_crop)

        with torch.no_grad():  # validation
            ms_up = Variable(ms_up_crop)
            ms = Variable(ms_org_crop)
            pan = Variable(pan_crop)
            if torch.cuda.is_available():
                model.cuda()
                ms_up = ms_up.cuda()
                ms = ms.cuda()
                pan = pan.cuda()
            # out = netG(z)  # 生成图片
            # out = model(pan_crop, ms_gray_crop)
            start = time.time()
            ### start testing ######
            out = overlap_crop_forward(ms_up, ms, pan, shave=16, min_size=271000, bic=None)  # 62000, or crop 1/4 :72000, crop 1/2 :271000
            # out = model(ms_up, ms, pan)

            end = time.time()
        output = out.cpu()

        time_ave = (end - start) / batch_size
        print('Average testing time is', time_ave)

        for i in range(batch_size):
            # image = (output.data[i] + 1) / 2.0
            count += 1
            image = output.data[i]
            # image = image.mul(255).byte()
            image = np.transpose(image.numpy(), (1, 2, 0))

            if sate == 'wv3_8':
                save_f_name = sate + '_%03d.mat' % count
            else:
                save_f_name = sate + '_%03d.tif' % count
            save_image_full(sate, save_f_name, image)

            # print(image.shape)
            # im_rgb = image[:, :, 0:3]
            # image = Image.fromarray(image)    # transfer to pillow image
            # im_rgb = Image.fromarray(im_rgb)
            # im_rgb.show()
            # image.save(os.path.join(image_path, '%d_out_tf.tif' % (file_name.data[i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--satellite', type=str, default='pl')  # satellite here
    parser.add_argument('--checkpoint', type=str, default='ik_model_epoch_500.pth')
    parser.add_argument('--dataset_dir', type=str, default='E:\\remote sense image fusion\\Source Images\\')
    parser.add_argument('--ratio', type=int, default=4)  # ratio here
    parser.add_argument("--net", default='FusionNet')
    parser.add_argument('--testBatchSize', type=int, default=15, help='testing batch size')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--cuda', default=False, help='use cuda?')
    opt = parser.parse_args()

    model = MSAN(1).cuda().eval()  ########## need to change

    # opt.satellite = 'ik'
    opt.satellite = 'wv3_8'

    if opt.satellite == 'ik':
        opt.checkpoint = 'ik_epoch_235.pth'
    elif opt.satellite == 'pl':
        opt.checkpoint = 'pl_epoch_235.pth'
    elif opt.satellite == 'wv3_8':
        opt.checkpoint = 'wv3_8_epoch_500.pth'

    dataset_dir = 'F:\\remote sense image fusion\\Source Images\\'
    model_path = r'.\model_trained\\'

    test_set = TestDatasetFromFolder_Full(dataset_dir, opt.satellite, upscale_factor=1)  # 测试集导入
    # test_set = get_test_set(opt.dataset_dir, opt.satellite, opt.ratio)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                  shuffle=False)
    checkpoint = torch.load(model_path + opt.satellite + '/%s' % opt.checkpoint,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    # model.eval()
    # r'E:\remote sense image fusion\compared_methods\2021-FusionNet\FusionNet-main'
    #                r'\FSnet_ik_pl_lu\models\%s' % opt.checkpoint,
    #                map_location=lambda storage, loc: storage)

    # image_path = r'../fused'
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)

    test(test_data_loader, model, opt.satellite)
