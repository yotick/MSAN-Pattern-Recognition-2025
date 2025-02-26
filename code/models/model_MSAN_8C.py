import copy
import math

import torch.nn.functional as F
from torch import nn
import torch

from models.swin_transf_lu_conv2 import swin

n_head = 4
in_size = 32
in_pixels = in_size ** 2
linear_dim = 64
n_feats = 30  # fixed
patch_size = 32


class MSAN(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(MSAN, self).__init__()

        self.swin1 = swin(
            # pretrain_img_size=128,
            in_chans=30,
            patch_size=4,
            embed_dim=90,
            # depths=[2, 2, 6, 2],
            # num_heads=[3, 6, 12, 24],
            depths=[2, 2],
            num_heads=[3, 6],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            # out_indices=(0, 1, 2, 3),
            out_indices=(0, 1),
            frozen_stages=-1,
            use_checkpoint=False,
            pretrained=None,
            init_cfg=None
        )
        self.blk_9_30_3 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(9, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blk_60_30 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.blk_30_60 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(30, 60, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.blk_60_30_1 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.blk_60_30_5 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=5, padding=2),
            nn.PReLU()
        )

        self.blk_60_30_7 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=7, padding=3),
            nn.PReLU()
        )
        self.blk_30_15_5 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(30, 15, kernel_size=5, padding=2),
            nn.PReLU()
        )

        self.blk_30_15_7 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(30, 15,  kernel_size=7, padding=3),
            nn.PReLU()
        )
        # self.up_sample4 = UpsampleBLock(8, 4)  ### for 4 band 4, for 8 band 8 ,in_channels, up_scale
        # self.up_sample2 = UpsampleBLock(4, 2)  ### for 4 band 4, for 8 band 8

        self.conv6 = nn.Conv2d(in_channels=30, out_channels=8, kernel_size=3, stride=1, padding=1,
                               bias=True)  # change out as 4   or   8



    #######################
    def forward(self, ms_up, ms_org, pan):
        # ms_org_up = self.up_sample4(ms_org)  ## in_channels, in_channels * up_scale ** 2
        # ms_up2 = self.up_sample2(ms_org)

        data1 = torch.cat([ms_up, pan], dim=1)

        mix_conv = self.blk_9_30_3(data1)

        out_s0, out_s1 = self.swin1(mix_conv)

        out1 = self.blk_30_60(mix_conv)

        out3_1 = self.blk_60_30_5(out1)
        out3_2 = self.blk_60_30_7(out1)

        # # out2 = torch.cat([out2_2, out2_1], dim=1)
        out3_s1 = out3_1 * out_s0
        out3_s2 = out3_2 * out_s0
        out3 = torch.cat([out3_s1, out3_s2], dim=1)
        out3 = out3 + out1

        out3_4 = self.blk_60_30(out3)
        #
        out4_1 = self.blk_30_15_5(out3_4)
        out4_2 = self.blk_30_15_7(out3_4)
        out4_s1 = out4_1 * out_s1
        out4_s2 = out4_2 * out_s1
        out4 = torch.cat([out4_s1, out4_s2], dim=1)
        out4 = out3_4 + out4


        out8 = self.conv6(out4)

        out_f = out8 + ms_up

        return out_f
        # return out_f, out2, out_d

