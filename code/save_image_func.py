from os import listdir
from os.path import join
# import cv2 as cv
import numpy as np
import scipy.io as sio
from PIL import Image
import torchvision.utils as utils
from torchvision.transforms import ToTensor
import torch


def save_image_RS(sate, file_name, fused):

    fused_rgb = fused[:, :, 0:3]
    fused_nir = fused[:, :, 3]
    # fused = np.uint8(fused * 255)

    if sate == 'wv3_8':
        save_path_fused = join(r'fused/WorldView-3-8/wv3-8b', file_name)
        save_path_rgb = join('fused/WorldView-3-8/wv3-8-rgb', ''.join([file_name[:-4], '.tif']))
        # save_path_ms_up = join(r'../fused\WorldView-3-8\wv3-8-up', file_name)

        fused = ToTensor()(fused)
        ndarr = fused.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        sio.savemat(save_path_fused, {'fused': ndarr})

    if sate == 'ik':
        save_path_fused = join(r'fused/IKONOS/ik_4b', file_name)
        save_path_rgb = join('fused/IKONOS/ik_rgb', ''.join([file_name[:-4], '.tif']))
        # save_path_ms_up = join(r'./fused\IKONO\ik_up_ms', file_name)

    if sate == 'pl':
        # save_path_fused = join(r'fused/Pleiades/pl_4b', file_name)
        # save_path_rgb = join('fused/Pleiades/pl_rgb', ''.join([file_name[:-4], '.tif']))

        save_path_fused = join(r'fused/Pleiades_ik/pl_4b', file_name)    ####  need to change here#################
        save_path_rgb = join('fused/Pleiades_ik/pl_rgb', ''.join([file_name[:-4], '.tif']))

        # save_path_fused = join(r'fused/Pleiades_noSwin/pl_4b', file_name)
        # save_path_rgb = join('fused/Pleiades_noSwin/pl_rgb', ''.join([file_name[:-4], '.tif']))

        # save_path_ms_up = join(r'./fused\Pleiades\pl_up_ms',file_name)

    if sate == 'ik' or sate == 'pl':

        fused = ToTensor()(fused)
        utils.save_image(fused, save_path_fused)

    fused_rgb = ToTensor()(fused_rgb)
    utils.save_image(fused_rgb, save_path_rgb)


def save_image_full(sate, file_name, fused):

    fused_rgb = fused[:, :, 0:3]
    fused_nir = fused[:, :, 3]
    # fused = np.uint8(fused * 255)

    if sate == 'wv3_8':
        save_path_fused = join(r'fused/WorldView-3-8/wv3-8b-FS', file_name)
        save_path_rgb = join('fused/WorldView-3-8/wv3-8-rgb-FS', ''.join([file_name[:-4], '.tif']))
        # save_path_ms_up = join(r'../fused\WorldView-3-8\wv3-8-up', file_name)

        fused = ToTensor()(fused)
        ndarr = fused.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        sio.savemat(save_path_fused, {'fused': ndarr})

    if sate == 'ik':
        save_path_fused = join(r'fused/IKONOS/ik_4b_FS', file_name)
        save_path_rgb = join('fused/IKONOS/ik_rgb_FS', ''.join([file_name[:-4], '.tif']))
        # save_path_ms_up = join(r'./fused\IKONO\ik_up_ms', file_name)

    if sate == 'pl':
        save_path_fused = join(r'fused/Pleiades/pl_4b_FS', file_name)
        save_path_rgb = join('fused/Pleiades/pl_rgb_FS', ''.join([file_name[:-4], '.tif']))
        # save_path_ms_up = join(r'./fused\Pleiades\pl_up_ms',file_name)

    if sate == 'ik' or sate == 'pl':

        fused = ToTensor()(fused)
        utils.save_image(fused, save_path_fused)

    fused_rgb = ToTensor()(fused_rgb)
    utils.save_image(fused_rgb, save_path_rgb)
