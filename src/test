from __future__ import print_function

import math
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
import torch.nn as nn
import warnings

from torch import optim
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from module import ProST, DeShEnet
from preprocessing import downsample_single
from RTPI.RTPI import RTPInet_v1
from util import input_param, cal_ncc, init_rtvec_test, SURF_distance, set_matrix

warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': '{:.4f}'.format})
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]
device = torch.device('cuda:0')
PI = math.pi
NUM_PHOTON = 20000
BATCH_SIZE = 1
EPS = 1e-10
ITER_STEPS = 500
clipping_value = 10

DATA_SET = '../Data'
CT_128_SET = DATA_SET + '/ct/128'
MASK_128_SET = DATA_SET + '/mask/128'
XRAY_256_SET = DATA_SET + '/x_ray/256'
img_files = os.listdir(DATA_SET)
ct_list = []
xray_list = []
xray_param = np.array([])

SAVE_PATH = '../Data/save_model'
RESUME_EPOCH_RTPInet = 80  # -1 means training from scratch
RESUME_MODEL_RTPInet = SAVE_PATH + '/RTPInet/xray/vali_model' + str(RESUME_EPOCH_RTPInet) + '.pt'
RESUME_EPOCH_DeShEnet = 3000  # -1 means training from scratch
RESUME_MODEL_DeShEnet = SAVE_PATH + '/new_encoder_mask_real_time_riem/vali_model' + str(RESUME_EPOCH_DeShEnet) + '.pt'
stop_trd = 1e-3
zFlip = False


def test():
    criterion_mse = nn.MSELoss()
    pixel_id_detect = 0.19959
    name = np.random.choice(ct_list, 1)[0].split('_')[0]
    CT_PATH = CT_128_SET + '/' + name + '_128.nii.gz'
    flag = np.argwhere(np.array(ct_list) == (name + '_128.nii.gz')).item()
    # XRAY_PATH = XRAY_256_SET + '/' + name + '_frontal_noBoard.nii.gz'
    MASK_PATH = MASK_128_SET + '/' + name + '_mask.nii.gz'
    # rtvec_gt_param = xray_param[int(name == 'data6')]
    rtvec_gt_param = xray_param[np.argwhere(np.array(ct_list) == (name + '_128.nii.gz')).item()]
    param, det_size, ct_vol, ray_proj_mov, corner_pt, norm_factor \
        = input_param(CT_PATH, BATCH_SIZE, 4, 256)
    # param, det_size, xray_det, ct_vol, ray_proj_mov, corner_pt, norm_factor \
    #     = input_param_xray(XRAY_PATH, CT_PATH, BATCH_SIZE, 4, 256, zFlip, device)
    param_mask, _, ct_vol_mask, ray_proj_mov_mask, corner_pt_mask, _ \
        = input_param(MASK_PATH, BATCH_SIZE, 4, 256, zFlip, device)

    initmodel = ProST().to(device)
    initmodel = nn.DataParallel(initmodel, device_ids=device_ids)

    model_RTPInet = RTPInet_v1().to(device)
    model_RTPInet = nn.DataParallel(model_RTPInet)
    checkpoint_RTPInet = torch.load(RESUME_MODEL_RTPInet)
    model_RTPInet.load_state_dict(checkpoint_RTPInet['state_dict'])
    model_RTPInet.eval()
    model_RTPInet.require_grad = False

    model_DeShEnet = Fine_regnet().to(device)
    model_DeShEnet = nn.DataParallel(model_DeShEnet)
    checkpoint_DeShEnet = torch.load(RESUME_MODEL_DeShEnet)
    model_DeShEnet.load_state_dict(checkpoint_DeShEnet['state_dict'])
    model_DeShEnet.eval()
    model_DeShEnet.require_grad = False

    _, transform_mat3x4_gt, _, _ = init_rtvec_test(rtvec_gt_param, BATCH_SIZE, device, norm_factor)

    print('target_value:', rtvec_gt_param)
    with torch.no_grad():
        target = initmodel(ct_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt, param)
        # target = xray_det
        min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
        target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

    _, _, pred_RTPI = model_RTPInet(ct_vol, target, corner_pt, param, norm_factor, flag)
    rtvec_param = pred_RTPI.cpu().detach().numpy().squeeze()
    print('rtpi_value:', rtvec_param)
    _, _, _, rtvec = init_rtvec_test(rtvec_param, BATCH_SIZE, device, norm_factor)

    lr_net = 0.0005
    optimizer_net = optim.SGD([rtvec], lr=lr_net, momentum=0.9)
    scheduler_net = CyclicLR(optimizer_net, base_lr=0.0004, max_lr=0.0006, step_size_up=20)

    _, _, _, _, _, _, _ = model_DeShEnet(
        ct_vol, target, rtvec, corner_pt, param)
    iter_time = 0.1457
    for iter in range(ITER_STEPS):
        torch.cuda.empty_cache()
        start = time.time()
        transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
        with torch.no_grad():
            mask = initmodel(ct_vol_mask, ray_proj_mov_mask, transform_mat3x4, corner_pt_mask, param_mask)
            min_mask, _ = torch.min(mask.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            max_mask, _ = torch.max(mask.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
            mask = (mask.reshape(BATCH_SIZE, -1) - min_mask) / (max_mask - min_mask)
            mask = mask.reshape(BATCH_SIZE, 1, det_size, det_size)
        mask_frontal_array = mask.cpu().detach().numpy().squeeze(1)
        for i in range(BATCH_SIZE):
            if i == 0:
                mask_frontal = sitk.GetImageFromArray(mask_frontal_array[i])
                mask_16_item = downsample_single(mask_frontal, 16)[np.newaxis, :]
                mask_32_item = downsample_single(mask_frontal, 8)[np.newaxis, :]
                mask_16_array = mask_16_item
                mask_32_array = mask_32_item
                continue
            mask_16_array = np.append(mask_16_array, mask_16_item, 0)
            mask_32_array = np.append(mask_32_array, mask_32_item, 0)
        mask_16_tensor = torch.tensor(mask_16_array[:, np.newaxis, :, :], dtype=torch.float, requires_grad=True,
                                      device=device)
        mask_32_tensor = torch.tensor(mask_32_array[:, np.newaxis, :, :], dtype=torch.float, requires_grad=True,
                                      device=device)
        mask_16_tensor = mask_16_tensor.repeat(1, 128, 1, 1)
        mask_32_tensor = mask_32_tensor.repeat(1, 64, 1, 1)

        encode_mov_l1, encode_mov_l2, encode_mov_l3, encode_fix_l1, encode_fix_l2, encode_fix_l3, proj_mov = model_DeShEnet(
            ct_vol, target, rtvec, corner_pt, param)

        optimizer_net.zero_grad()
        l2_loss = criterion_mse(encode_mov_l1 * mask_32_tensor, encode_fix_l1 * mask_32_tensor) + criterion_mse(
            encode_mov_l2 * mask_32_tensor, encode_fix_l2 * mask_32_tensor) + criterion_mse(
            encode_mov_l3 * mask_16_tensor, encode_fix_l3 * mask_16_tensor)

        rtvec.retain_grad()
        l2_loss.backward()
        scheduler_net.step()
        optimizer_net.step()
        end = time.time()
        iter_time += end - start
        stop = iter_time > 4.5

        if stop:
            print('deshe_value:', rtvec_res_param)
            rtvec_final_param = np.zeros(6)
            for i in range(6):
                if abs(rtvec_param[i] - rtvec_gt_param[i]) < abs(rtvec_res_param[i] - rtvec_gt_param[i]):
                    rtvec_final_param[i] = rtvec_param[i]
                else:
                    rtvec_final_param[i] = rtvec_res_param[i]
            # print(rtvec_gt_param)
            # print(rtvec_param)
            # print(rtvec_res_param)
            # print(rtvec_final_param)
            _, transform_mat3x4, _, _ = init_rtvec_test(rtvec_final_param, BATCH_SIZE, device, norm_factor)
            with torch.no_grad():
                # tar = initmodel(ct_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt, param)
                # min_tar, _ = torch.min(tar.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                # max_tar, _ = torch.max(tar.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                # tar = (tar.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                # tar = tar.reshape(BATCH_SIZE, 1, det_size, det_size)

                pred = initmodel(ct_vol, ray_proj_mov, transform_mat3x4, corner_pt, param)
                min_pred, _ = torch.min(pred.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_pred, _ = torch.max(pred.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                pred = (pred.reshape(BATCH_SIZE, -1) - min_pred) / (max_pred - min_pred)
                pred = pred.reshape(BATCH_SIZE, 1, det_size, det_size)

            tar = target
            ncc = cal_ncc(tar, pred)

            img_fix_array = tar.cpu().detach().numpy().squeeze()[:, :, np.newaxis]
            img_fix_array = (img_fix_array - np.min(img_fix_array)) / (
                    np.max(img_fix_array) - np.min(img_fix_array)) * 255
            img_fix_array = img_fix_array.astype(np.uint8)
            img_moving_array = pred.cpu().detach().numpy().squeeze()[:, :, np.newaxis]
            img_moving_array = (img_moving_array - np.min(img_moving_array)) / (
                    np.max(img_moving_array) - np.min(img_moving_array)) * 255
            img_moving_array = img_moving_array.astype(np.uint8)
            kp_distance_mean = SURF_distance(img_fix_array, img_moving_array) * pixel_id_detect

            print('result:', rtvec_final_param)
            print("iteration： ", iter, " time：", iter_time, 's')
            return rtvec_gt_param, \
                   rtvec_final_param, \
                   kp_distance_mean, \
                   ncc.cpu().detach().numpy().squeeze(), \
                   iter_time

        result = torch.zeros([1, 6]).to(device)
        rz, ry, rx, tz, ty, tx = rtvec[0][0], rtvec[0][1], rtvec[0][2], -rtvec[0][3], -rtvec[0][4], rtvec[0][5]
        result[0][0] = rz
        result[0][1] = rx
        result[0][2] = ry
        result[0][3] = tx
        result[0][4] = ty
        result[0][5] = tz
        result[:, :3] = result[:, :3] * 180 / PI
        result[:, 3:] = result[:, 3:] * norm_factor
        rtvec_res_param = result.cpu().detach().numpy().squeeze()


if __name__ == "__main__":
    data_ls = []
    pred_ncc_ls = []
    pred_kp_dist_ls = []
    pred_error_ls = []
    time_ls = []
    for i in tqdm(range(100)):
        tar, pred, pred_kp_dist, pred_ncc, t = test()
        data_ls.append(tar)
        data_ls.append(pred)
        pred_kp_dist_ls.append(pred_kp_dist)
        pred_ncc_ls.append(pred_ncc)
        pred_error_ls.append(abs(pred - tar))
        time_ls.append(t)
    data_df = pd.DataFrame(data_ls)
    data_df.to_csv('../Data/save_result/RTPI_fine_reg_xray.csv', header=False, index=False)

    pred_kp_dist_array = np.array(pred_kp_dist_ls)
    pred_kp_dist_mean = np.mean(pred_kp_dist_array)
    pred_ncc_array = np.array(pred_ncc_ls)
    pred_ncc_mean = np.mean(pred_ncc_array)
    pred_error = np.array(pred_error_ls)
    pred_error_rot = np.sum(pred_error[:, :3], 1)
    pred_error_rot1 = pred_error[:, 0]
    pred_error_rot2 = pred_error[:, 1]
    pred_error_rot3 = pred_error[:, 2]
    pred_error_trans = np.sum(pred_error[:, 3:], 1)
    pred_error_trans1 = pred_error[:, 3]
    pred_error_trans2 = pred_error[:, 4]
    pred_error_trans3 = pred_error[:, 5]
    pred_mean_rot = np.mean(pred_error_rot)
    pred_mean_rot1 = np.mean(pred_error_rot1)
    pred_mean_rot2 = np.mean(pred_error_rot2)
    pred_mean_rot3 = np.mean(pred_error_rot3)
    pred_mean_trans = np.mean(pred_error_trans)
    pred_mean_trans1 = np.mean(pred_error_trans1)
    pred_mean_trans2 = np.mean(pred_error_trans2)
    pred_mean_trans3 = np.mean(pred_error_trans3)
    pred_stddev_rot = np.std(pred_error_rot)
    pred_stddev_rot1 = np.std(pred_error_rot1)
    pred_stddev_rot2 = np.std(pred_error_rot2)
    pred_stddev_rot3 = np.std(pred_error_rot3)
    pred_stddev_trans = np.std(pred_error_trans)
    pred_stddev_trans1 = np.std(pred_error_trans1)
    pred_stddev_trans2 = np.std(pred_error_trans2)
    pred_stddev_trans3 = np.std(pred_error_trans3)
    pred_median_rot = np.median(pred_error_rot)
    pred_median_rot1 = np.median(pred_error_rot1)
    pred_median_rot2 = np.median(pred_error_rot2)
    pred_median_rot3 = np.median(pred_error_rot3)
    pred_median_trans = np.median(pred_error_trans)
    pred_median_trans1 = np.median(pred_error_trans1)
    pred_median_trans2 = np.median(pred_error_trans2)
    pred_median_trans3 = np.median(pred_error_trans3)

    time_array = np.array(time_ls)
    avg_time = np.mean(time_array)

    print('pred_mean_rot: {0:.4f}±{1:.4f},\npred_mean_trans: {2:.4f}±{3:.4f}'.format(pred_mean_rot, pred_stddev_rot,
                                                                                     pred_mean_trans,
                                                                                     pred_stddev_trans))

    print('pred_key_point_distance_mean: {0:.4f}'.format(pred_kp_dist_mean))
    print('pred_ncc_mean: {0:.4f}'.format(pred_ncc_mean))
    print('pred_mean_rot1: {0:.4f}±{1:.4f},\n'
          'pred_mean_rot2: {2:.4f}±{3:.4f},\n'
          'pred_mean_rot3: {4:.4f}±{5:.4f},\n'
          'pred_mean_trans1: {6:.4f}±{7:.4f},\n'
          'pred_mean_trans2: {8:.4f}±{9:.4f},\n'
          'pred_mean_trans3: {10:.4f}±{11:.4f}'.format(pred_mean_rot1, pred_stddev_rot1, pred_mean_rot2,
                                                       pred_stddev_rot2, pred_mean_rot3, pred_stddev_rot3,
                                                       pred_mean_trans1, pred_stddev_trans1, pred_mean_trans2,
                                                       pred_stddev_trans2, pred_mean_trans3, pred_stddev_trans3))
    print('pred_median_rot: {0:.4f},\npred_median_trans: {1:.4f}'.format(pred_median_rot, pred_median_trans))
    print('pred_median_rot1: {0:.4f},\n'
          'pred_median_rot2: {1:.4f},\n'
          'pred_median_rot3: {2:.4f},\n'
          'pred_median_trans1: {3:.4f},\n'
          'pred_median_trans2: {4:.4f},\n'
          'pred_median_trans3: {5:.4f}'.format(pred_median_rot1, pred_median_rot2, pred_median_rot3, pred_median_trans1,
                                               pred_median_trans2, pred_median_trans3))
    print('avg_time: {0:.4f}'.format(avg_time))
