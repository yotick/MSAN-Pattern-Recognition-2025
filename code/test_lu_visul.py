import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from data_set_py.data_utils_RS import TestDatasetFromFolder

from models.model_MSAN_4C import MSAN  ################ need to change ######################
# from models.model_8_swin_good import Generator

from save_image_func import save_image_RS
import time
from torch import nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 指定要测试的数据样本的索引位置
test_index = 35


def test(test_data_loader, model, sate):
    val_bar = tqdm(test_data_loader)  # 验证集的进度条
    count = 0
    for ms_up_crop, ms_org_crop, pan_crop, gt_crop in val_bar:
        batch_size = ms_up_crop.size(0)  # current batch size
        # valing_results['batch_sizes'] += batch_size
        # detail_crop = detail_crop.type(torch.FloatTensor)  # to make the type the same as model
        # data = torch.cat(ms_up_crop, pan_crop)

        ###############  new add, save middle variable #################################
        # 定义一个中间输出结果列表
        # outputs = []
        #
        # # ################ 在模型的第二个卷积层注册一个Hook
        # def hook1(module, input, output):  ## for convolution layer
        #     # output_tensor = torch.cat(output, dim=1)
        #     outputs.append(output.clone().detach())
        #
        # def hook2(module, input, output):  ## for swin layer
        #     output_tensor = torch.cat(output, dim=1)
        #     outputs.append(output_tensor.clone().detach())
        #
        # # model.swin1.register_forward_hook(hook2)  # swin layer
        # model.blk_30_15_3.register_forward_hook(hook1)  ####### need to change!!!
        # #############  change end ##################


        ########### 查看某层输入 #############
        # 创建Hook对象，用于获取某一层的输入
        # class InputHook:
        #     def __init__(self):
        #         self.inputs = []
        #
        #     def __call__(self, module, input, output):
        #         self.inputs.append(input[0].detach())
        #
        # hook3 = InputHook()
        # # 注册Hook函数
        # handle = model.conv6.register_forward_hook(hook3)
        ########### 查看某层输入end #############

        out4_results = []  # 创建一个空列表来保存out4的输出结果

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
            out = model(ms_up, ms, pan)
            end = time.time()
        output = out.cpu()

        # ########## 输出中间输出结果的完整张量信息 ########################
        # for i, output in enumerate(outputs):
        #     torch.save(output, 'visualize\\noMul_blk15_{}'.format(test_index) + '.pt')  #### need to change here#####
        # ############ change end  #####################

        ############# 获取某一层的输入 ########################
        # input = hook3.inputs[0]
        # torch.save(input, 'visualize\\noSwin_conv6_{}'.format(test_index) + '.pt')    #### need to change here#####
        ########  end ###############

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
            save_image_RS(sate, save_f_name, image)

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
    parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--cuda', default=False, help='use cuda?')
    opt = parser.parse_args()

    model = MSAN(1).cuda().eval()

    # opt.satellite = 'wv3_8'
    opt.satellite = 'pl'

    if opt.satellite == 'ik':
        opt.checkpoint = 'ik_epoch_270.pth'
        # opt.checkpoint = 'ik_epoch_270_light.pth'
    elif opt.satellite == 'pl':            #####　need to change here#############
        opt.checkpoint = 'pl_epoch_280_good.pth'
        # opt.checkpoint = 'pl_epoch_280_light.pth'
        # opt.checkpoint = 'pl_epoch_200_noSwin.pth'
        # opt.checkpoint = 'pl_epoch_200_noMul.pth'
    elif opt.satellite == 'wv3_8':
        # opt.checkpoint = 'wv3_8_epoch_300_light.pth'
        opt.checkpoint = 'wv3_8_epoch_300_good.pth'

    dataset_dir = 'E:\\remote sense image fusion\\Source Images\\'
    model_path = r'.\model_trained\\'

    test_set = TestDatasetFromFolder(dataset_dir, opt.satellite, upscale_factor=1)  # 测试集导入
    # test_set = get_test_set(opt.dataset_dir, opt.satellite, opt.ratio)
    ##############  added ###############################

    # 使用Subset类创建新的数据集对象，包含指定索引位置的数据样本
    test_sample = Subset(test_set, [test_index])
    ##############  add end, should change Dataloader below ###########

    test_data_loader = DataLoader(dataset=test_sample, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                  shuffle=False)
    checkpoint = torch.load(model_path + opt.satellite + '/%s' % opt.checkpoint,
                            map_location=lambda storage, loc: storage)
    # model = nn.DataParallel(model)
    model.load_state_dict(checkpoint, False)
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    # model.eval()
    # r'E:\remote sense image fusion\compared_methods\2021-FusionNet\FusionNet-main'
    #                r'\FSnet_ik_pl_lu\models\%s' % opt.checkpoint,
    #                map_location=lambda storage, loc: storage)

    # image_path = r'../fused'
    # if not os.path.exists(image_path):
    #     os.makedirs(image_path)

    test(test_data_loader, model, opt.satellite)
