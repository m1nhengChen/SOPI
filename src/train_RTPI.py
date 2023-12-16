from __future__ import print_function
from util import seed_everything
seed = 772512  # 182501, 852097, 881411, 772512
print('seed:', seed)
seed_everything(seed)

import geomstats.geometry.riemannian_metric as riem
import math
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from torch.optim.lr_scheduler import CyclicLR

from metric import ncc, pixel_wise_error
from module import ProST, ProST_drr_generator
from RTPI import RTPInet_v3
from util import input_param, count_parameters, init_rtvec, pose2rtvec, rtvec2pose, early_stop, domain_randomization

warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': '{:.4f}'.format})

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))

PI = math.pi
NUM_PHOTON = 20000
BATCH_SIZE = 1
EPS = 1e-10
ITER_NUM = 200
EPOCH_NUM = 1000
EARLY_STOP_THRESHOLD = 20
EARLY_STOP_COUNTER = 0
clipping_value = 10
SAVE_MODEL_EVERY_EPOCH = 5

use_gd = True
SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
RiemMetric = RiemannianMetric(dim=6)
METRIC = SE3_GROUP.left_canonical_metric
riem_dist_fun = RiemMetric.dist

use_val = False
version = 3
TRAIN_SET = '../original_data/ct/256/train'
train_img_files = os.listdir(TRAIN_SET)
train_ct_list = []
for train_img_file in train_img_files:
    train_ct_list.append(train_img_file)
if use_val:
    VAL_SET = '../original_data/ct/256/train'
    val_img_files = os.listdir(VAL_SET)
    val_ct_list = []
    for val_img_file in val_img_files:
        val_ct_list.append(val_img_file)

use_ncc = False
use_pwe = False
proj_size = 256
flag = 2
max_lr = 1e-3
base_lr = max_lr * 1e-2 #1e-3
weight_l2_norm_rot =
weight_l2_norm_trans =
weight_l2_norm_trans1 =
weight_l2_norm_trans23 =
weight_l2_norm =
weight_ncc =
weight_pwe =
mse = nn.MSELoss()

if use_ncc:
    SAVE_PATH = f'../Data/save_model/RTPI/v{str(version)}/with_ncc'
elif use_pwe:
    SAVE_PATH = f'../Data/save_model/RTPI/v{str(version)}/with_pwe'
elif use_gd:
    SAVE_PATH = f'../Data/save_model/RTPI/v{str(version)}/gd'
else:
    SAVE_PATH = f'../Data/save_model/RTPI/v{str(version)}/l2'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
model_list = os.listdir(SAVE_PATH)
model_num = len(model_list)
RESUME_EPOCH = -1
is_resume = False
if is_resume:
    RESUME_EPOCH = (model_num - 1) * 5
RESUME_MODEL = f'{SAVE_PATH}/vali_model{str(RESUME_EPOCH)}.pt'
print(RESUME_MODEL)

zFlip = False
accumulation_steps = 4
use_accumulation = True


def train():
    # seed=random.randint(0,1000000)
    np.random.shuffle(train_ct_list)
    train_set = train_ct_list
    if use_val:
        np.random.shuffle(val_ct_list)
        val_set = val_ct_list

    # using the parallel training
    use_multi_gpu = True
    initmodel = ProST().to(device)
    if version == 3:
        model = RTPInet_v3().to(device)
    elif version == 4:
        model = RTPInet_v4().to(device)
    elif version == 5:
        model = RTPInet_v5().to(device)
    if use_multi_gpu:
        initmodel = nn.DataParallel(initmodel, device_ids=device_ids)
        model = nn.DataParallel(model, device_ids=device_ids)

    # train_loss = val_loss = 0
    train_loss_list = []
    train_dist_loss_list = []
    if use_ncc:
        train_ncc_loss_list = []
    elif use_pwe:
        train_pwe_loss_list = []
    train_total_loss_list = []
    train_mse_loss_list = []
    if use_val:
        val_loss_list = []
        val_dist_loss_list = []
        if use_ncc:
            val_ncc_loss_list = []
        elif use_pwe:
            val_pwe_loss_list = []
        val_total_loss_list = []
        val_mse_loss_list = []

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=100, cycle_momentum=False)
    # scheduler = ReduceLROnPlateau(optimizer, scheduler, val_loss)

    if RESUME_EPOCH >= 0:
        print('Resuming model from epoch', RESUME_EPOCH)
        checkpoint = torch.load(RESUME_MODEL)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        START_EPOCH = RESUME_EPOCH + 1
        step_cnt = RESUME_EPOCH * ITER_NUM
    else:
        START_EPOCH = 0
        step_cnt = 0

    print('module parameters %.2fM' % (count_parameters(model) / 1e6))
    model.train()

    for epoch in range(START_EPOCH, START_EPOCH + EPOCH_NUM + 1):
        # Do Iterative Validation
        print('********************************************************************************')
        model.train()
        for iter in range(ITER_NUM):
            torch.cuda.empty_cache()
            # train
            train_ct = np.random.choice(train_set, 1)[0]
            # train_name = train_ct.split('_')[0]
            TRAIN_CT_PATH = TRAIN_SET + '/' + train_ct
            # TRAIN_CT_PATH = '../Data/ct/256/data5_256.nii.gz'
            train_param, train_det_size, train_ct_vol, train_ray_proj_mov, train_corner_pt, train_norm_factor \
                = input_param(TRAIN_CT_PATH, BATCH_SIZE, flag, proj_size)
            _, train_transform_mat3x4_gt, _, train_rtvec_gt = init_rtvec(BATCH_SIZE, device, train_norm_factor, center=[90., 0., 0., 900., 0., 0.], distribution='N')

            train_tar = rtvec2pose(train_rtvec_gt, train_norm_factor, device)
            train_rtvec_gt_param = train_tar.cpu().detach().numpy().squeeze()

            step_cnt = step_cnt + 1
            scheduler.step()
            
            with torch.no_grad():
                train_target = initmodel(train_ct_vol, train_ray_proj_mov, train_transform_mat3x4_gt, train_corner_pt,
                                         train_param)
                train_min_tar, _ = torch.min(train_target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                train_max_tar, _ = torch.max(train_target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                train_target = (train_target.reshape(BATCH_SIZE, -1) - train_min_tar) / (train_max_tar - train_min_tar)
                train_target = train_target.reshape(BATCH_SIZE, 1, train_det_size, train_det_size)
                train_target = domain_randomization(train_target, device)

            if version == 3:
                train_pred = model(train_ct_vol, train_target, train_norm_factor)
            elif version == 4:
                train_pred = model(train_target, train_norm_factor)
            elif version == 5:
                train_pred = model(train_ct_vol, train_target)
            # return

            train_rtvec_param = train_pred.cpu().detach().numpy().squeeze()
            train_rtvec = pose2rtvec(train_pred, device, train_norm_factor)

            if use_gd:
                train_dist_loss = riem.loss(train_rtvec, train_rtvec_gt, METRIC)
            else:
                train_dist_loss = (weight_l2_norm_rot * torch.norm(train_rtvec[:, :3] - train_rtvec_gt[:, :3]) \
                                    + weight_l2_norm_trans * torch.norm(train_rtvec[:, 3:] - train_rtvec_gt[:, 3:])) / 6
            # train_l2_norm_loss = (weight_l2_norm_rot * torch.norm(train_rtvec[:, :3] - train_rtvec_gt[:, :3]) \
            #                       + weight_l2_norm_trans1 * torch.norm(train_rtvec[:, 3] - train_rtvec_gt[:, 3]) \
            #                       + weight_l2_norm_trans23 * torch.norm(train_rtvec[:, 4:] - train_rtvec_gt[:, 4:])) / 6
            train_mse_loss = mse(train_tar, train_pred)
            if use_ncc:
                train_proj_mov = ProST_drr_generator(train_rtvec, train_ct_vol, train_ray_proj_mov, BATCH_SIZE,
                                                     train_param, train_det_size, train_corner_pt, train_norm_factor)
                train_ncc_loss = ncc(train_target, train_proj_mov)
                if use_gd:
                    train_total_loss = train_dist_loss
                else:
                    train_total_loss = weight_l2_norm * train_dist_loss + weight_ncc * train_ncc_loss
            elif use_pwe:
                train_proj_mov = ProST_drr_generator(train_rtvec, train_ct_vol, train_ray_proj_mov, BATCH_SIZE,
                                                     train_param, train_det_size, train_corner_pt, train_norm_factor)
                train_pwe_loss = pixel_wise_error(train_target, train_proj_mov)
                if use_gd:
                    train_total_loss = train_dist_loss
                else:
                    train_total_loss = weight_l2_norm * train_dist_loss + weight_pwe * train_pwe_loss
            else:
                if use_gd:
                    train_total_loss = train_dist_loss
                else:
                    train_total_loss = weight_l2_norm * train_dist_loss
            # return
            
            if use_val:
                # val
                val_ct = np.random.choice(val_set, 1)[0]
                # val_name = val_ct.split('_')[0]
                VAL_CT_PATH = VAL_SET + '/' + val_ct
                val_param, val_det_size, val_ct_vol, val_ray_proj_mov, val_corner_pt, val_norm_factor \
                    = input_param(VAL_CT_PATH, BATCH_SIZE, flag, proj_size)
                _, val_transform_mat3x4_gt, _, val_rtvec_gt = init_rtvec(BATCH_SIZE, device, val_norm_factor)

                val_tar = rtvec2pose(val_rtvec_gt, val_norm_factor, device)
                val_rtvec_gt_param = val_tar.cpu().detach().numpy().squeeze()

                with torch.no_grad():
                    val_target = initmodel(val_ct_vol, val_ray_proj_mov, val_transform_mat3x4_gt, val_corner_pt, val_param)
                    val_target = domain_randomization(val_target, device)
                    val_min_tar, _ = torch.min(val_target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    val_max_tar, _ = torch.max(val_target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                    val_target = (val_target.reshape(BATCH_SIZE, -1) - val_min_tar) / (val_max_tar - val_min_tar)
                    val_target = val_target.reshape(BATCH_SIZE, 1, val_det_size, val_det_size)

                if version == 3:
                    val_pred = model(val_ct_vol, val_target, val_norm_factor)
                elif version == 4:
                    val_pred = model(val_target, val_norm_factor)
                elif version == 5:
                    val_pred = model(val_ct_vol, val_target)

                val_rtvec_param = val_pred.cpu().detach().numpy().squeeze()
                val_rtvec = pose2rtvec(val_pred, device, val_norm_factor)

                if use_gd:
                    val_dist_loss = riem.loss(val_rtvec, val_rtvec_gt, METRIC)
                else:
                    val_dist_loss = (weight_l2_norm_rot * torch.norm(val_rtvec[:, :3] - val_rtvec_gt[:, :3]) \
                                        + weight_l2_norm_trans * torch.norm(val_rtvec[:, 3:] - val_rtvec_gt[:, 3:])) / 6
                # val_l2_norm_loss = (weight_l2_norm_rot * torch.norm(val_rtvec[:, :3] - val_rtvec_gt[:, :3]) \
                #                       + weight_l2_norm_trans1 * torch.norm(val_rtvec[:, 3] - val_rtvec_gt[:, 3]) \
                #                       + weight_l2_norm_trans23 * torch.norm(val_rtvec[:, 4:] - val_rtvec_gt[:, 4:])) / 6
                val_mse_loss = mse(val_tar, val_pred)
                if use_ncc:
                    val_proj_mov = ProST_drr_generator(val_rtvec, val_ct_vol, val_ray_proj_mov, BATCH_SIZE, val_param,
                                                    val_det_size, val_corner_pt, val_norm_factor)
                    val_ncc_loss = ncc(val_target, val_proj_mov)
                    if use_gd:
                        val_total_loss = val_dist_loss
                    else:
                        val_total_loss = weight_l2_norm * val_dist_loss + weight_ncc * val_ncc_loss
                elif use_pwe:
                    val_proj_mov = ProST_drr_generator(val_rtvec, val_ct_vol, val_ray_proj_mov, BATCH_SIZE, val_param,
                                                    val_det_size, val_corner_pt, val_norm_factor)
                    val_pwe_loss = pixel_wise_error(val_target, val_proj_mov)
                    if use_gd:
                        val_total_loss = val_dist_loss
                    else:
                        val_total_loss = weight_l2_norm * val_dist_loss + weight_pwe * val_pwe_loss
                else:
                    if use_gd:
                        val_total_loss = val_dist_loss
                    else:
                        val_total_loss = weight_l2_norm * val_dist_loss

            if use_accumulation:
                # loss regularization
                train_loss = train_total_loss / accumulation_steps

                # back propagation
                train_loss.backward()

                # update parameters of net
                if (iter + 1) % accumulation_steps == 0:
                    # optimizer the net
                    optimizer.step()  # update parameters of net
                    optimizer.zero_grad()  # reset gradient
                    torch.cuda.empty_cache()

                train_dist_loss_list.append(torch.mean(train_dist_loss).detach().item())
                train_total_loss_list.append(train_total_loss.detach().item())
                train_mse_loss_list.append(train_mse_loss.detach().item())
                if use_val:
                    val_dist_loss_list.append(torch.mean(val_dist_loss).detach().item())
                    val_total_loss_list.append(val_total_loss.detach().item())
                    val_mse_loss_list.append(val_mse_loss.detach().item())

                cur_lr = float(scheduler.get_lr()[0])
            else:
                optimizer.zero_grad()  # reset gradient
                train_total_loss.backward()
                optimizer.step()

                train_dist_loss_list.append(torch.mean(train_dist_loss).detach().item())
                train_total_loss_list.append(train_total_loss.detach().item())
                train_mse_loss_list.append(train_mse_loss.detach().item())
                if use_val:
                    val_dist_loss_list.append(torch.mean(val_dist_loss).detach().item())
                    val_total_loss_list.append(val_total_loss.detach().item())
                    val_mse_loss_list.append(val_mse_loss.detach().item())

                cur_lr = float(scheduler.get_lr()[0])
            # print('train_target_value:', train_rtvec_gt_param)
            # print('train_rtpi_value', train_rtvec_param)
            # if use_val:
            #     print('val_target_value:', val_rtvec_gt_param)
            #     print('val_rtpi_value', val_rtvec_param)
            # print('--------------------------------------------------------------------------------')
        train_loss = np.mean(train_total_loss_list)
        train_mse_loss_mean = np.mean(train_mse_loss_list)
        if use_val:
            val_loss = np.mean(val_total_loss_list)
            val_mse_loss_mean = np.mean(val_mse_loss_list)
        if use_ncc:
            train_ncc_loss_list.append(train_ncc_loss.detach().item())
            if use_val:
                val_ncc_loss_list.append(val_ncc_loss.detach().item())
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}, train_nccLoss: {:.4f}/{:.2f}, \nval_tLoss: {:.4f}, val_dLoss: {:.4f}/{:.2f}, val_nccLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list),
                        np.mean(train_ncc_loss_list), np.std(train_ncc_loss_list), \
                        val_loss, np.mean(val_dist_loss_list), np.std(val_dist_loss_list), np.mean(val_ncc_loss_list),
                        np.std(val_ncc_loss_list)))
            else:
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}, train_nccLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list),
                        np.mean(train_ncc_loss_list), np.std(train_ncc_loss_list)))
        elif use_pwe:
            train_pwe_loss_list.append(train_pwe_loss.detach().item())
            if use_val:
                val_pwe_loss_list.append(val_pwe_loss.detach().item())
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}, train_pweLoss: {:.4f}/{:.2f}, \nval_tLoss: {:.4f}, val_dLoss: {:.4f}/{:.2f}, val_pweLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list),
                        np.mean(train_pwe_loss_list), np.std(train_pwe_loss_list), \
                        val_loss, np.mean(val_dist_loss_list), np.std(val_dist_loss_list), np.mean(val_pwe_loss_list),
                        np.std(val_pwe_loss_list)))
            else:
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}, train_pweLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list),
                        np.mean(train_pwe_loss_list), np.std(train_pwe_loss_list)))
        else:
            if use_val:
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}, \nval_tLoss: {:.4f}, val_dLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list), \
                        val_loss, np.mean(val_dist_loss_list), np.std(val_dist_loss_list)))
            else:
                print(
                    'Train epoch: {}, Iter: {}, LR: {:.4f}, \ntrain_tLoss: {:.4f}, train_dLoss: {:.4f}/{:.2f}'.format(
                        epoch, iter, cur_lr, \
                        train_loss, np.mean(train_dist_loss_list), np.std(train_dist_loss_list)))

        if epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, SAVE_PATH + f'/vali_model{str(epoch)}.pt')

        train_loss_list.append(train_mse_loss_mean)
        if use_val:
            val_loss_list.append(val_mse_loss_mean)

        train_dist_loss_list = []
        if use_ncc:
            train_ncc_loss_list = []
        elif use_pwe:
            train_pwe_loss_list = []
        train_total_loss_list = []
        train_mse_loss_list = []
        if use_val:
            val_dist_loss_list = []
            if use_ncc:
                val_ncc_loss_list = []
            elif use_pwe:
                val_pwe_loss_list = []
            val_total_loss_list = []
            val_mse_loss_list = []

        if use_val:
            if early_stop(val_mse_loss_mean, val_loss_list, EARLY_STOP_COUNTER, EARLY_STOP_THRESHOLD):
                print('Early stop!')
                return
    print('********************************************************************************')


if __name__ == "__main__":
    train()
