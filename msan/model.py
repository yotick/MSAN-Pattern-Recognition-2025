import torch
import torch.nn as nn
import torch.nn.functional as F

from .msan_swin import swin


class MSAN(nn.Module):
    """
    Standalone MSAN model.
    Interface keeps (spectral_num, spatial_num), forward(ms_up, pan).
    """

    def __init__(self, spectral_num: int = 8, spatial_num: int = 1):
        super().__init__()
        self.spectral_num = spectral_num
        self.spatial_num = spatial_num

        if spatial_num != 1:
            raise ValueError(
                f"Unsupported MSAN setting: spatial_num={spatial_num}. Original MSAN expects PAN channel=1."
            )

        in_ch = spectral_num + spatial_num
        out_ch = spectral_num

        self.swin1 = swin(
            in_chans=30,
            patch_size=4,
            embed_dim=90,
            depths=[2, 2],
            num_heads=[3, 6],
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1),
            frozen_stages=-1,
            use_checkpoint=False,
            pretrained=None,
            init_cfg=None,
        )

        self.blk_in_30_3 = nn.Sequential(nn.Conv2d(in_ch, 30, kernel_size=3, padding=1), nn.PReLU())
        self.blk_60_30 = nn.Sequential(nn.Conv2d(60, 30, kernel_size=3, padding=1), nn.PReLU())
        self.blk_30_60 = nn.Sequential(nn.Conv2d(30, 60, kernel_size=3, padding=1), nn.PReLU())
        self.blk_60_30_5 = nn.Sequential(nn.Conv2d(60, 30, kernel_size=5, padding=2), nn.PReLU())
        self.blk_60_30_7 = nn.Sequential(nn.Conv2d(60, 30, kernel_size=7, padding=3), nn.PReLU())
        self.blk_30_15_5 = nn.Sequential(nn.Conv2d(30, 15, kernel_size=5, padding=2), nn.PReLU())
        self.blk_30_15_7 = nn.Sequential(nn.Conv2d(30, 15, kernel_size=7, padding=3), nn.PReLU())
        self.conv6 = nn.Conv2d(30, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, ms_up, pan):
        data1 = torch.cat([ms_up, pan], dim=1)
        mix_conv = self.blk_in_30_3(data1)

        h, w = mix_conv.shape[-2], mix_conv.shape[-1]
        if (h, w) != (128, 128):
            swin_in = F.interpolate(mix_conv, size=(128, 128), mode="bilinear", align_corners=False)
            out_s0, out_s1 = self.swin1(swin_in)
            out_s0 = F.interpolate(out_s0, size=(h, w), mode="bilinear", align_corners=False)
            out_s1 = F.interpolate(out_s1, size=(h, w), mode="bilinear", align_corners=False)
        else:
            out_s0, out_s1 = self.swin1(mix_conv)

        out1 = self.blk_30_60(mix_conv)
        out3_1 = self.blk_60_30_5(out1)
        out3_2 = self.blk_60_30_7(out1)
        out3 = torch.cat([out3_1 * out_s0, out3_2 * out_s0], dim=1)
        out3 = out3 + out1

        out3_4 = self.blk_60_30(out3)
        out4_1 = self.blk_30_15_5(out3_4)
        out4_2 = self.blk_30_15_7(out3_4)
        out4 = torch.cat([out4_1 * out_s1, out4_2 * out_s1], dim=1)
        out4 = out3_4 + out4

        out = self.conv6(out4)
        return out + ms_up

