from __future__ import print_function

import geomstats.geometry.riemannian_metric as riem
import math
import numpy as np
import os
import SimpleITK as sitk
import sys
import torch
import torch.nn as nn
import warnings

from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR

from module import ProST, Composite_encoder
from preprocessing import downsample_single
from util import input_param, init_rtvec_train

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_ids = [0]
device = torch.device('cuda:0')
# matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")
PI = math.pi
NUM_PHOTON = 20000
BATCH_SIZE = 1
EPS = 1e-10
ITER_NUM = 200
clipping_value = 10
SAVE_MODEL_EVERY_EPOCH = 1

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
RiemMetric = RiemannianMetric(dim=6)
METRIC = SE3_GROUP.left_canonical_metric
riem_dist_fun = RiemMetric.dist

USER_DIR = '/home/chenminheng'  # 'D:/about_my_college_life/Academic/SRTP' #'/home/cmh'
DATA_SET = USER_DIR + '/ProSTGrid/Data'
CT_128_SET = DATA_SET + '/ct/128/all'
# MASK_SET = DATA_SET + '/mask_frontal'
MASK_128_SET = DATA_SET + '/mask/128'
img_files = os.listdir(CT_128_SET)
ct_list = []
for img_file in img_files:
    if img_file.startswith('CTJ') and 'unmask' not in img_file and img_file.endswith('.nii.gz'):
        ct_list.append(img_file[:-17])
SAVE_PATH = '../Data/save_model'

RESUME_EPOCH = 0  # -1 means training from scratch
RESUME_MODEL = SAVE_PATH + '/new_encoder_mask_real_time_riem/vali_model' + str(RESUME_EPOCH) + '.pt'

zFlip = False


def train():
    criterion_mse = nn.MSELoss()

    use_multi_gpu = True

    if use_multi_gpu:
        initmodel = ProST().to(device)
        initmodel = nn.DataParallel(initmodel, device_ids=device_ids)
        model = Composite_encoder().to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        initmodel = ProST().to(device)
        model = Composite_encoder().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, step_size_up=100)

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

    total = sum([param.nelement() for param in model.parameters()])
    print("module parameters %.2fM" % (total / 1e6))

    model.train()

    riem_grad_loss_list = []
    riem_grad_rot_loss_list = []
    riem_grad_trans_loss_list = []
    riem_dist_list = []
    riem_dist_mean_list = []
    mse_loss_list = []
    vecgrad_diff_list = []
    total_loss_list = []

    for epoch in range(START_EPOCH, 20000):
        ## Do Iterative Validation
        model.train()
        for iter in range(ITER_NUM):
            torch.cuda.empty_cache()
            CT_NAME = np.random.choice(ct_list, 1)[0]
            CT_128_PATH = CT_128_SET + '/' + CT_NAME + '_resampled.nii.gz'
            # MASK_16_PATH = MASK_SET + '/16/' + CT_NAME + '_mask.nii.gz'
            # MASK_32_PATH = MASK_SET + '/32/' + CT_NAME + '_mask.nii.gz'
            MASK_128_PATH = MASK_128_SET + '/' + CT_NAME + '_mask.nii.gz'
            # mask_16 = sitk.ReadImage(MASK_16_PATH)
            # mask_32 = sitk.ReadImage(MASK_32_PATH)
            # mask_16_array = sitk.GetArrayFromImage(mask_16)
            # mask_32_array = sitk.GetArrayFromImage(mask_32)
            # mask_16_tensor = torch.tensor(mask_16_array, dtype=torch.float, requires_grad=True, device=device)
            # mask_32_tensor = torch.tensor(mask_32_array, dtype=torch.float, requires_grad=True, device=device)
            param, det_size, ct_vol, ray_proj_mov, corner_pt, norm_factor \
                = input_param(CT_128_PATH, BATCH_SIZE, 4, 256, zFlip, device)

            param_mask, _, ct_vol_mask, ray_proj_mov_mask, corner_pt_mask, _ \
                = input_param(MASK_128_PATH, BATCH_SIZE, 4, 256, zFlip, device)
            step_cnt = step_cnt + 1
            scheduler.step()
            # Get target  projection
            transform_mat3x4, transform_mat3x4_gt, rtvec, rtvec_gt, _, _ \
                = init_rtvec_train(BATCH_SIZE, device, norm_factor)

            target = torch.zeros((BATCH_SIZE, 1, det_size, det_size), device=device)
            # print(str(slice_num))
            with torch.no_grad():
                target = initmodel(ct_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt, param)
                min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

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

            # Do Projection and get two encodings
            encode_mov_l1, encode_mov_l2, encode_mov_l3, encode_fix_l1, encode_fix_l2, encode_fix_l3, proj_mov = model(
                ct_vol, target, rtvec, corner_pt, param)
            # Calculate Net l2 Loss, L_N
            l2_loss = criterion_mse(encode_mov_l1 * mask_32_tensor, encode_fix_l1 * mask_32_tensor) + criterion_mse(
                encode_mov_l2 * mask_32_tensor, encode_fix_l2 * mask_32_tensor) + criterion_mse(
                encode_mov_l3 * mask_16_tensor, encode_fix_l3 * mask_16_tensor)

            # Find geodesic distance
            riem_dist = np.sqrt(riem.loss(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC))

            z = Variable(torch.ones(l2_loss.shape)).to(device)
            rtvec_grad = torch.autograd.grad(l2_loss, rtvec, grad_outputs=z, only_inputs=True, create_graph=True,
                                             retain_graph=True)[0]
            # Find geodesic gradient
            riem_grad = riem.grad(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC)
            riem_grad = torch.tensor(riem_grad, dtype=torch.float, requires_grad=False, device=device)

            ### Translation Loss
            riem_grad_transnorm = riem_grad[:, 3:] / (torch.norm(riem_grad[:, 3:], dim=-1, keepdim=True) + EPS)
            rtvec_grad_transnorm = rtvec_grad[:, 3:] / (torch.norm(rtvec_grad[:, 3:], dim=-1, keepdim=True) + EPS)
            riem_grad_trans_loss = torch.mean(torch.sum((riem_grad_transnorm - rtvec_grad_transnorm) ** 2, dim=-1))

            ### Rotation Loss
            riem_grad_rotnorm = riem_grad[:, :3] / (torch.norm(riem_grad[:, :3], dim=-1, keepdim=True) + EPS)
            rtvec_grad_rotnorm = rtvec_grad[:, :3] / (torch.norm(rtvec_grad[:, :3], dim=-1, keepdim=True) + EPS)
            riem_grad_rot_loss = torch.mean(torch.sum((riem_grad_rotnorm - rtvec_grad_rotnorm) ** 2, dim=-1))

            riem_grad_loss = riem_grad_trans_loss + riem_grad_rot_loss

            loss_isnan = True in torch.isnan(riem_grad_loss)
            if loss_isnan:
                loss_isnan_idx = torch.where(riem_grad_loss != riem_grad_loss)
                print(loss_isnan_idx)

            assert torch.isnan(riem_grad_loss).sum() == 0, print(riem_grad_loss)

            optimizer.zero_grad()

            riem_grad_loss.backward()

            # loss_isnan = True in torch.isnan(l2_loss)
            # if loss_isnan:
            #     loss_isnan_idx = torch.where(l2_loss != l2_loss)
            #     print(loss_isnan_idx)

            # assert torch.isnan(l2_loss).sum() == 0, print(l2_loss)

            # optimizer.zero_grad()

            # l2_loss.backward()
            # print("grad before clip:"+str(model.linear.weight.grad))
            # nn.utils.clip_grad_norm(model.parameters, 10, norm_type=2)
            # print("grad after clip:"+str(model.linear.weight.grad))
            # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

            optimizer.step()

            total_loss = l2_loss

            mse_loss_list.append(torch.mean(l2_loss).detach().item())
            riem_grad_loss_list.append(riem_grad_loss.detach().item())
            riem_grad_rot_loss_list.append(riem_grad_rot_loss.detach().item())
            riem_grad_trans_loss_list.append(riem_grad_trans_loss.detach().item())
            riem_dist_list.append(riem_dist)
            riem_dist_mean_list.append(torch.mean(riem_dist))
            total_loss_list.append(total_loss.detach().item())
            vecgrad_diff = (rtvec_grad - riem_grad).detach().cpu().numpy()
            vecgrad_diff_list.append(vecgrad_diff)

            torch.cuda.empty_cache()

            cur_lr = float(scheduler.get_lr()[0])

            print(
                'Train epoch: {} Iter: {} tLoss: {:.4f}, gLoss: {:.4f}/{:.2f}, gLoss_rot: {:.4f}/{:.2f}, gLoss_trans: {:.4f}/{:.2f}, LR: {:.4f}'.format(
                    epoch, iter, np.mean(total_loss_list), np.mean(riem_grad_loss_list), np.std(riem_grad_loss_list), \
                    np.mean(riem_grad_rot_loss_list), np.std(riem_grad_rot_loss_list), \
                    np.mean(riem_grad_trans_loss_list), np.std(riem_grad_trans_loss_list),
                    cur_lr, sys.stdout))
            # print(
            #     'Train epoch: {} Iter: {} tLoss: {:.4f}, LR: {:.4f}'.format(
            #         epoch, iter, np.mean(total_loss_list), cur_lr, sys.stdout))

        if epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, SAVE_PATH + '/new_encoder_mask_real_time_riem/vali_model' + str(epoch) + '.pt')

        riem_grad_loss_list = []
        riem_grad_rot_loss_list = []
        riem_grad_trans_loss_list = []
        riem_dist_list = []
        riem_dist_mean_list = []
        mse_loss_list = []
        vecgrad_diff_list = []


if __name__ == "__main__":
    train()
