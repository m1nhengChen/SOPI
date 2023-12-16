from __future__ import print_function

import geomstats.geometry.riemannian_metric as riem
import math
import numpy as np
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

from module import ProST, Fine_regnet
from util import input_param, count_parameters, init_rtvec, seed_everything, domain_randomization, rtvec2pose

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
PI = math.pi
NUM_PHOTON = 20000
BATCH_SIZE = 2
EPS = 1e-10
ITER_NUM = 200
clipping_value = 10
SAVE_MODEL_EVERY_EPOCH = 5

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
RiemMetric = RiemannianMetric(dim=6)
METRIC = SE3_GROUP.left_canonical_metric
riem_dist_fun = RiemMetric.dist

RAW_SET = '../../original_data/ct/128/train'
SEG_SET = '../../Data/ct/128/train'
img_files = os.listdir(SEG_SET)
ct_list = []
for img_file in img_files:
    ct_list.append(img_file)
# SAVE_PATH = '../../Data/save_model/ProST/frontal_bsl'
SAVE_PATH = '../../Data/save_model/SOPI/fine_reg_bsl'
if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

RESUME_EPOCH = 105 # -1 means training from scratch
RESUME_MODEL = SAVE_PATH + '/vali_model' + str(RESUME_EPOCH) + '.pt'

zFlip = False
flag = 4
proj_size = 256


def train():
    # seed = random.randint(0,1000000)
    seed = 645632 # 781723 611780
    print('seed:', seed)
    seed = seed_everything(seed)
    criterion_mse = nn.MSELoss()

    use_multi_gpu = True

    if use_multi_gpu:
        initmodel = ProST().to(device)
        initmodel = nn.DataParallel(initmodel, device_ids=device_ids)
        model = Fine_regnet().to(device)
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        initmodel = ProST().to(device)
        model = Fine_regnet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4, step_size_up=100)

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

    print('module parameters:', count_parameters(model))

    model.train()

    riem_grad_loss_list = []
    riem_grad_rot_loss_list = []
    riem_grad_trans_loss_list = []
    riem_dist_list = []
    riem_dist_mean_list = []
    mse_loss_list = []
    vecgrad_diff_list = []
    total_loss_list = []
    # print('1:', torch.cuda.memory_allocated()/1024/1024/1024)
    for epoch in range(START_EPOCH, 20000):
        # Do Iterative Validation
        model.train()
        for iter in range(ITER_NUM):
            torch.cuda.empty_cache()
            CT_NAME = np.random.choice(ct_list, 1)[0]
            RAW_PATH = RAW_SET + '/' + CT_NAME.split('_')[0] + '.nii.gz'
            SEG_PATH = SEG_SET + '/' + CT_NAME
            _, _, RAW_vol, _, _, _ \
                = input_param(RAW_PATH, BATCH_SIZE, flag, proj_size, device=device)
            param, det_size, SEG_vol, ray_proj_mov, corner_pt, norm_factor \
                = input_param(SEG_PATH, BATCH_SIZE, flag, proj_size, device=device)
            step_cnt = step_cnt + 1
            scheduler.step()
            # Get target projection
            _, transform_mat3x4_gt, rtvec, rtvec_gt \
                = init_rtvec(BATCH_SIZE, device, norm_factor, center=[90, 0, 0, 900, 0, 0], distribution='N', iterative=True)
            # print('target: ', rtvec2pose(rtvec_gt, norm_factor, device).detach().cpu().numpy())
            # print('initial: ', rtvec2pose(rtvec, norm_factor, device).detach().cpu().numpy())
            with torch.no_grad():
                target = initmodel(RAW_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt, param)
                min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
                # target = domain_randomization(target, device)
            # print('2:', torch.cuda.memory_allocated()/1024/1024/1024)
            # Do Projection and get two encodings
            _, _, encode_mov, _, _, encode_tar, _ = model(SEG_vol, target, rtvec, corner_pt, param)
            # print('3:', torch.cuda.memory_allocated()/1024/1024/1024)
            # print('result: ', rtvec2pose(rtvec, norm_factor, device).detach().cpu().numpy())

            optimizer.zero_grad()
            # Calculate Net l2 Loss, L_N
            l2_loss = criterion_mse(encode_mov, encode_tar)

            # Find geodesic distance
            riem_dist = np.sqrt(riem.loss(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC))

            z = Variable(torch.ones(l2_loss.shape)).to(device)
            rtvec_grad = torch.autograd.grad(l2_loss, rtvec, grad_outputs=z, only_inputs=True, create_graph=True,
                                             retain_graph=True)[0]
            # Find geodesic gradient
            riem_grad = riem.grad(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC)
            riem_grad = torch.tensor(riem_grad, dtype=torch.float, requires_grad=False, device=device)

            # Translation Loss
            riem_grad_transnorm = riem_grad[:, 3:] / (torch.norm(riem_grad[:, 3:], dim=-1, keepdim=True) + EPS)
            rtvec_grad_transnorm = rtvec_grad[:, 3:] / (torch.norm(rtvec_grad[:, 3:], dim=-1, keepdim=True) + EPS)
            riem_grad_trans_loss = torch.mean(torch.sum((riem_grad_transnorm - rtvec_grad_transnorm) ** 2, dim=-1))

            # Rotation Loss
            riem_grad_rotnorm = riem_grad[:, :3] / (torch.norm(riem_grad[:, :3], dim=-1, keepdim=True) + EPS)
            rtvec_grad_rotnorm = rtvec_grad[:, :3] / (torch.norm(rtvec_grad[:, :3], dim=-1, keepdim=True) + EPS)
            riem_grad_rot_loss = torch.mean(torch.sum((riem_grad_rotnorm - rtvec_grad_rotnorm) ** 2, dim=-1))

            riem_grad_loss = riem_grad_trans_loss + riem_grad_rot_loss
            # print('4:', torch.cuda.memory_allocated()/1024/1024/1024)

            riem_grad_loss.backward()
            # print('5:', torch.cuda.memory_allocated()/1024/1024/1024)

            # Clip training gradient magnitude
            # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            total_loss = l2_loss

            mse_loss_list.append(torch.mean(l2_loss).detach().item())
            riem_grad_loss_list.append(riem_grad_loss.detach().item())
            riem_grad_rot_loss_list.append(riem_grad_rot_loss.detach().item())
            riem_grad_trans_loss_list.append(riem_grad_trans_loss.detach().item())
            riem_dist_list.append(riem_dist)
            riem_dist_mean_list.append(np.mean(riem_dist))
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
        #     'Train epoch: {} Iter: {} tLoss: {:.4f}, gLoss: {:.4f}/{:.2f}, gLoss_rot: {:.4f}/{:.2f}, gLoss_trans: {:.4f}/{:.2f}, LR: {:.4f}'.format(
        #         epoch, iter, np.mean(total_loss_list), np.mean(riem_grad_loss_list), np.std(riem_grad_loss_list), \
        #         np.mean(riem_grad_rot_loss_list), np.std(riem_grad_rot_loss_list), \
        #         np.mean(riem_grad_trans_loss_list), np.std(riem_grad_trans_loss_list),
        #         cur_lr, sys.stdout))
            # print('6:', torch.cuda.memory_allocated()/1024/1024/1024)
            # print('max:',torch.cuda.max_memory_allocated()/1024/1024/1024)

        if epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, SAVE_PATH + '/vali_model' + str(epoch) + '.pt')

        riem_grad_loss_list = []
        riem_grad_rot_loss_list = []
        riem_grad_trans_loss_list = []
        riem_dist_list = []
        riem_dist_mean_list = []
        mse_loss_list = []
        vecgrad_diff_list = []


if __name__ == "__main__":
    train()
