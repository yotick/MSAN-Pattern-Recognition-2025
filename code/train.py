import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch.optim as op
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
# import tqdm
from torch import nn
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from data_set_py.data_utils_RS import TrainDatasetFromFolder, ValDatasetFromFolder

from models.model_MSAN_4C import MSAN
# from models.model_fusformer import MainNet
# from model_8_R3A1_notgoo import Generator
from img_index import ref_evaluate
import pandas as pd
from datetime import datetime
from Pansharpening_Toolbox_Assessment_Python.indexes_evaluation import indexes_evaluation
import time
from helpers import make_patches

############# need to cahnge herer!!!!
sate = 'pl'
# sate = 'wv3_8'
dataset_dir = 'E:\\remote sense image fusion\\Source Images\\'

patch_size = 32
val_step = 5

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=1, type=int, choices=[1, 2, 4],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')  # for 8 band 500
parser.add_argument('--batch_size', default=32, type=int, help='train epoch number')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning Rate. Default=0.01')  # for 8 band 0.0006, for 4 band half
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by "
                                                          "momentum every n epochs, Default: n=500")
parser.add_argument("--log_path", type=str, default="training_results\\")

opt = parser.parse_args()

CROP_SIZE = opt.crop_size  # 裁剪会带来拼尽问题嘛
UPSCALE_FACTOR = opt.upscale_factor  # 上采样
NUM_EPOCHS = opt.num_epochs  # 轮数
BATCH_SIZE = opt.batch_size

train_set = TrainDatasetFromFolder(dataset_dir, sate, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)  # 训练集导入
val_set = ValDatasetFromFolder(dataset_dir, sate, upscale_factor=UPSCALE_FACTOR)  # 测试集导入
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)  # 训练集制作
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=20, shuffle=False)


# ================== Pre-Define =================== #
SEED = 15
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

device = torch.device('cuda:0')
model = MSAN(UPSCALE_FACTOR).to(device)


optimizerG = op.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    if lr < 0.00005:
        lr = 0.00005
    return lr


## added part
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


# optimizerD = op.Adam(netD.parameters())

#### mse_loss, tv_loss, lap_loss, total_loss ####
# results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
results = {'time': [], 'epoch': [], 'lr': [], 'mse_loss': [], 'l1_loss_d': [], 'l1_loss': [], 'total_loss': [],
           'psnr': [],
           'sam': [],
           'ergas': [], 'scc': [], 'q_avg': [], 'q2n': []}
# df = pd.DataFrame(results)
out_path = 'training_results/'  # 输出路径
# df.to_csv(out_path + sate + '_training_result.csv')

# val_results = {'mse_loss': [], 'tv_loss': [], 'lap_loss': [], 'total_loss': []}
writer = SummaryWriter()
count = 0

lr = opt.lr
# for epoch in range(1, NUM_EPOCHS + 1):

#########  start from saved models ########################
log_dir = ' '
# log_dir = './model_trained/pl/netG_pl_epoch_1_1230.pth'  # 模型保存路径
# log_dir = './model_trained/ik/netG_ik_epoch_1_1132.pth'
# log_dir = './model_trained/wv3_8/wv3_8_epoch_070.pth'
#
## 如果有保存的模型，则加载模型，并在其基础上继续训练
if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir)
    model.load_state_dict(checkpoint)
    # optimizerG.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']
    start_epoch = 70
    print('加载 epoch  成功！')
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')

t = time.strftime("%Y%m%d%H%M")

for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    train_bar = tqdm(train_loader)  # 进度条

    # print("epoch =", epoch, "lr =", optimizerG.param_groups[0]["lr"])

    lr = adjust_learning_rate(optimizerG, epoch)  ##### need to chagne here!!!

    for param_group in optimizerG.param_groups:
        param_group["lr"] = lr
    #
    # Adversarial Loss
    # running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0, 'g_adversarial_loss': 0,
    #                    'g_image_loss': 0, 'g_tv_loss': 0, 'g_ir_tv_loss': 0, 'g_d_perception_loss': 0}
    running_results = {'mse_loss': 0, 'l1_loss_d': 0, 'l1_loss': 0, 'total_loss': 0, 'batch_sizes': 0}

    model.train()
    # model.eval()
    for ms_up_crop, ms_org_crop, pan_crop, gt_crop, _, _ in train_bar:
        g_update_first = True
        batch_size = ms_up_crop.size(0)
        running_results['batch_sizes'] += batch_size  # pytorch batch_size 不一定一致

        ms_up = Variable(ms_up_crop)
        ms_org = Variable(ms_org_crop)
        pan = Variable(pan_crop)
        gt = Variable(gt_crop)
        if torch.cuda.is_available():
            ms_up = ms_up.cuda()
            ms_org = ms_org.cuda()
            pan = pan.cuda()
            gt = gt.cuda()

        out = model(ms_up, ms_org, pan)
        # out,_, _ = model(ms_org, pan)

        ##### testing #################
        # out_rgb = out[0, 0:3, :,:]
        # out2_rgb = out2[0, 0:3, :,:]
        # out_d_rgb = out2_d[0, 0:3, :,:]
        # ms_org_rgb = ms_org_crop[0, 0:3, :,:]
        #
        # out_pil = ToPILImage()(out_rgb)
        # out2_pil = ToPILImage()(out2_rgb)
        # out_d_pil = ToPILImage()(out_d_rgb)
        # ms_org_pil = ToPILImage()(ms_org_rgb)
        #
        # plt.subplot(141)
        # plt.imshow(out_pil)
        # plt.subplot(142)
        # plt.imshow(out2_pil)
        # plt.subplot(143)
        # plt.imshow(out_d_pil)
        # plt.subplot(144)
        # plt.imshow(ms_org_pil)
        # plt.show()

        out_image = out
        target = gt

        # target, unfold_shape = make_patches(target, patch_size=patch_size)

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        optimizerG.zero_grad()  # change
        l1 = nn.L1Loss()
        l1_loss = l1(out_image, target)

        mse = nn.MSELoss()
        mse_loss = mse(out_image, target)

        smoothl1 = nn.SmoothL1Loss()
        sml1_loss = smoothl1(out_image, target)
        # total_loss = mse_loss+0.2*l1_loss
        total_loss = l1_loss

        # mse_loss, l1_loss_d, l1_loss, total_loss = generator_criterion(out_image, target, out2_d, z2)
        total_loss.requires_grad_(True)
        total_loss.backward()
        # optimizerD.step()
        optimizerG.step()

        #### mse_loss, tv_loss, lap_loss, total_loss ####
        running_results['mse_loss'] += mse_loss.item() * batch_size
        # running_results['l1_loss_d'] += l1_loss_d.item() * batch_size
        running_results['l1_loss'] += l1_loss.item() * batch_size
        running_results['total_loss'] += total_loss.item() * batch_size

        train_bar.set_description(desc='lr:%f [%d/%d] mse_loss: %.5f l1_loss: %.5f total_loss: %.5f' % (
            lr, epoch, NUM_EPOCHS, running_results['mse_loss'] / running_results['batch_sizes'],
            running_results['l1_loss'] / running_results['batch_sizes'],
            running_results['total_loss'] / running_results['batch_sizes']))
        writer.add_scalar('mse_loss', running_results['mse_loss'] / running_results['batch_sizes'], count)
        # writer.add_scalar('l1_loss_d', running_results['l1_loss_d'] / running_results['batch_sizes'], count)
        writer.add_scalar('l1_loss', running_results['l1_loss'] / running_results['batch_sizes'], count)
        writer.add_scalar('total_loss', running_results['total_loss'] / running_results['batch_sizes'], count)
        # writer.add_scalar('g_image_loss', running_results['g_image_loss'] / running_results['batch_sizes'], count)
        # writer.add_scalar('g_adversarial_loss', running_results['g_adversarial_loss'] / running_results['batch_sizes'],
        #                   count)
        # writer.add_scalar('g_tv_loss', running_results['g_tv_loss'] / running_results['batch_sizes'], count)
        # writer.add_scalar('g_d_perception_loss',
        #                   running_results['g_d_perception_loss'] / running_results['batch_sizes'], count)
        count += 1
    model.eval()
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ############ 验证集 #################
    if epoch % val_step == 0:
        val_bar = tqdm(val_loader)  # 验证集的进度条
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}

        # for data, ms_gray_crop, pan_crop, ms_up_crop, gt_crop in train_bar:
        l_psnr = []
        l_sam = []
        l_ergas = []
        l_scc = []
        l_q = []
        l_q2n = []
        for ms_up_crop, ms_org_crop, pan_crop, gt_crop, _, _ in val_bar:
            batch_size = ms_up_crop.size(0)
            valing_results['batch_sizes'] += batch_size

            # gt_crop, unfold_shape2 = make_patches(gt_crop, patch_size=patch_size)
            # detail_crop = detail_crop.type(torch.FloatTensor)  # to make the type the same as model
            # data = torch.cat(ms_up_crop, pan_crop)
            with torch.no_grad():  # validation
                ms_up = Variable(ms_up_crop)
                ms_org = Variable(ms_org_crop)
                pan = Variable(pan_crop)
                if torch.cuda.is_available():
                    ms_up = ms_up.cuda()
                    ms_org = ms_org.cuda()
                    pan = pan.cuda()
                # out = netG(z)  # 生成图片
                # out = model(pan_crop, ms_gray_crop)
                # out, out2, out2_d = model(z1, z2, z3, sate)
                out = model(ms_up, ms_org, pan)

            output = out.cpu()
            count = 0
            for i in range(batch_size):
                val_images = []
                val_out = out.data[i].cpu().squeeze(0)
                val_gt0 = gt_crop.data[i].cpu().squeeze(0)
                # val_out = val_out.data.cpu().squeeze(0)
                # val_pan = val_pan.squeeze(0)
                # val_ms_gray = val_ms_gray.squeeze(0)

                # detail_gt_crop = detail_gt_crop * 2 - 0.5
                # val_out = val_out * 2 - 0.5

                val_fused = val_out
                val_gt = val_gt0

                val_rgb = val_fused[0:3]
                val_gt_rgb = val_gt[0:3]

                # val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])
                val_images.extend([val_rgb.squeeze(0), val_gt_rgb.squeeze(0)])

                # val_rgb = val_rgb.transpose(0, 2).transpose(0, 1).contiguous()

                # val_images.extend(
                #     [display_transform()(val_fused.squeeze(0)), display_transform()(val_gt.squeeze(0)),
                #      display_transform()(val_out.data.cpu().squeeze(0))])

                ##############  index evaluation ######################
                val_gt_np = val_gt.numpy().transpose(1, 2, 0)
                val_fused_np = val_fused.numpy().transpose(1, 2, 0)

                val_images = utils.make_grid(val_images, nrow=2, padding=5)
                utils.save_image(val_images, out_path + sate + '/images/' + sate + '_tensor_%d.tif' % i)

                [c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q] = ref_evaluate(val_fused_np, val_gt_np)
                # [Q2n_index, Q_index, ERGAS_index, SAM_index] = [0, 0, 0, 0]
                [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(val_fused_np, val_gt_np, 4, 8, 32, 1,
                                                                                  1,
                                                                                  1)
                l_psnr.append(c_psnr)
                l_sam.append(SAM_index)
                l_ergas.append(ERGAS_index)
                l_scc.append(c_scc)
                l_q.append(Q_index)
                l_q2n.append(Q2n_index)


        ##### finish val_bar ################
        psnr_avg = np.mean(l_psnr)
        sam_avg = np.mean(l_sam)
        ergas_avg = np.mean(l_ergas)
        scc_avg = np.mean(l_scc)
        q_avg = np.mean(l_q)
        q2n_avg = np.mean(l_q2n)

        print(
            'psnr:{:.4f}, sam:{:.4f}, ergas:{:.4f}, scc:{:.4f}, q:{:.4f},q2n:{:.4f}'.format(psnr_avg, sam_avg,
                                                                                            ergas_avg,
                                                                                            scc_avg, q_avg, q2n_avg))


        torch.save(model.state_dict(),
                   'model_trained/' + sate + '/' + sate + '_epoch_%03d.pth' % epoch)  # 存储网络参数
        #### save to excel  ####
        time_curr = "%s" % datetime.now()  # 获取当前时间
        results['time'].append(time_curr)
        results['epoch'].append(epoch)
        results['lr'].append(lr)
        results['mse_loss'].append(running_results['mse_loss'] / running_results['batch_sizes'])
        results['l1_loss_d'].append(running_results['l1_loss_d'] / running_results['batch_sizes'])
        results['l1_loss'].append(running_results['l1_loss'] / running_results['batch_sizes'])
        results['total_loss'].append(running_results['total_loss'] / running_results['batch_sizes'])
        results['psnr'].append(psnr_avg)
        results['sam'].append(sam_avg)
        results['ergas'].append(ergas_avg)
        results['scc'].append(scc_avg)
        results['q_avg'].append(q_avg)
        results['q2n'].append(q2n_avg)

        # train_log = open(os.path.join(opt.log_path, sate, "%s_%s_train.log") % (sate, t), "w")
        df = pd.DataFrame(results)  ###############################  need to change!!!
        df.to_excel(out_path + sate + '/' + sate + f'_{t}.xlsx', index=False)  #### need to change here!!!
writer.close()
