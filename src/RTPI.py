import math
import numpy as np
import ProSTGrid
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.parameter import Parameter

from module import Conv3dBlock, Conv2dBlock, Bottleneck, SplatBottleneck
from util import raydist_range, _bilinear_interpolate_no_torch_5D

PI = math.pi
device = torch.device('cuda')


class RTPInet_v1(nn.Module):
    '''
    Pose parameter Initialization of Rigid Transformation network
    the input of this network is  a  CT and a fluoroscopy image
    And this network is used to predict the initial parameter of a rigid transformation
    three rotation parameter(rx,ry,tz) and three translation parameter(tx,ty,tz)
    '''

    def __init__(self):
        super(RTPInet_v1, self).__init__()
        self.block3d1 = Conv3dBlock(1, 64, stride=1)
        self.block3d2 = Conv3dBlock(64, 32, stride=1)
        self.block3d3 = Conv3dBlock(32, 32, stride=1)
        self.fl3d = nn.Flatten(2, 4)
        self.block2d1 = Conv2dBlock(1, 64, stride=1)
        self.block2d2 = Conv2dBlock(64, 32, stride=1)
        self.localization = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          Bottleneck(32, 32),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(32, 16, 3, 1, 1),
                                          nn.ReLU(),
                                          Bottleneck(16, 16),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(16, 8, 3, 1, 1),
                                          nn.ReLU(),
                                          Bottleneck(8, 8),
                                          nn.MaxPool2d(2),
                                          )
        self.fl2d = nn.Flatten(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 6)
        self.drop1 = nn.Dropout2d(0.2)
        initial_transformation_params = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
        initial_transformation_params = torch.tensor(initial_transformation_params, device="cuda", requires_grad=True)
        self.fc2.bias = Parameter(initial_transformation_params)

    def forward(self, x, y, corner_pt, param, norm_factor, flag=-1):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]
        rx = self.block3d1(x)
        rx = self.block3d2(rx)
        rx = self.block3d3(rx)
        rx = self.fl3d(rx)
        rx = rx.view(-1, 32, 64, 64)
        ry = self.block2d1(y)
        ry = self.block2d2(ry)
        out = torch.cat([rx, ry], dim=1)
        out = self.localization(out)
        out = self.fl2d(out)
        out = F.relu(self.fc1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        # max_value = 1.0
        # Threshold = 0.0
        out = F.relu6(out)
        out = out / 6
        BATCH_SIZE = out.size()[0]
        H = y.size()[2]
        W = y.size()[3]
        # print('out:', out.cpu().detach().numpy().squeeze())
        if 0 <= flag < 3:
            angle_x_min, angle_x_max = 7 / 18 * PI, 11 / 18 * PI
            if flag == 0:
                angle_x_min = angle_x_max = (np.random.uniform(-1, 1, (1))[0] * 10 + 80) / 180 * PI
                x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 430
            rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
            angle_yz_min, angle_yz_max = -1 / 18 * PI, 1 / 18 * PI
            ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
            rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
            if flag == 1:
                x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 475
            if flag == 2:
                x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 430
            tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
            y_mov_min = y_mov_max = 20.1
            ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
            z_mov_min = z_mov_max = 10.5
            tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        elif 3 <= flag < 5:
            angle_x_min, angle_x_max = 6 / 18 * PI, 12 / 18 * PI
            rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
            angle_yz_min, angle_yz_max = -1 / 9 * PI, 1 / 9 * PI
            ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
            rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
            if flag == 3:
                x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 689.3
                z_mov_min = z_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 21.4
            if flag == 4:
                x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 + 642.5
                z_mov_min = z_mov_max = np.random.uniform(-1, 1, (1))[0] * 15 - 18.8
            tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
            y_mov_min = y_mov_max = 3.5
            ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
            tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        elif flag == 6:
            angle_x_min, angle_x_max = 6 / 18 * PI, 12 / 18 * PI
            rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
            angle_yz_min, angle_yz_max = -1 / 18 * PI, 1 / 18 * PI
            ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
            rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
            x_mov_min, x_mov_max = 600, 800
            tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
            y_mov_min, y_mov_max = -10, 10
            ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
            z_mov_min, z_mov_max = 10, 30
            tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        else:
            angle_x_min, angle_x_max = 7 / 18 * PI, 11 / 18 * PI
            rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
            angle_yz_min, angle_yz_max = -1 / 18 * PI, 1 / 18 * PI
            ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
            rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
            x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 30 + 700
            tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
            y_mov_min, y_mov_max = -30, 30
            ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
            z_mov_min, z_mov_max = -30, 30
            tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        # angle_x_min, angle_x_max = (rtvec_gt_param[0]-10)/ 180 * PI, (rtvec_gt_param[0]+20) / 180 * PI
        # rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
        # angle_y_min, angle_y_max = (rtvec_gt_param[1]-5)/ 180 * PI, (rtvec_gt_param[1]-10)/ 180 * PI
        # ry = out[:, 1] * (angle_y_max - angle_y_min) + angle_y_min
        # angle_z_min, angle_z_max = (rtvec_gt_param[2]-5)/ 180 * PI, (rtvec_gt_param[2]-10)/ 180 * PI
        # rz = out[:, 2] * (angle_z_max - angle_z_min) + angle_z_min
        # x_mov_min, x_mov_max = rtvec_gt_param[3]-30, rtvec_gt_param[3]+30
        # tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
        # y_mov_min, y_mov_max = rtvec_gt_param[4]-15, rtvec_gt_param[4]+15
        # ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
        # z_mov_min, z_mov_max = rtvec_gt_param[5]-15, rtvec_gt_param[5]+15
        # tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        pred = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        rtvec = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        pred[:, 0] = rx * 180 / PI
        pred[:, 1] = ry * 180 / PI
        pred[:, 2] = rz * 180 / PI
        pred[:, 3] = tx * norm_factor
        pred[:, 4] = ty * norm_factor
        pred[:, 5] = tz * norm_factor
        radian_x = rx
        radian_y = rz
        radian_z = ry
        x_mov = -tz
        y_mov = -ty
        z_mov = tx
        rtvec[:, 0] = radian_x
        rtvec[:, 1] = radian_y
        rtvec[:, 2] = radian_z
        rtvec[:, 3] = x_mov
        rtvec[:, 4] = y_mov
        rtvec[:, 5] = z_mov
        rotation_x = torch.cat(
            (torch.tensor([[1, 0, 0, 0]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1),
             torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.cos(radian_x).unsqueeze(1).unsqueeze(1),
                        -torch.sin(radian_x).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.sin(radian_x).unsqueeze(1).unsqueeze(1),
                        torch.cos(radian_x).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1)), 1)
        rotation_y = torch.cat(
            (torch.cat((torch.cos(radian_y).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.sin(radian_y).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.tensor([[0, 1, 0, 0]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1),

             torch.cat((-torch.sin(radian_y).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.cos(radian_y).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1)), 1)
        rotation_z = torch.cat(
            (torch.cat((torch.cos(radian_z).unsqueeze(1).unsqueeze(1),
                        -torch.sin(radian_z).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.cat((torch.sin(radian_z).unsqueeze(1).unsqueeze(1),
                        torch.cos(radian_z).unsqueeze(1).unsqueeze(1),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device)), 2),
             torch.tensor([[0, 0, 1, 0]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1),
             torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1)), 1)
        trans_mat = torch.cat(
            (torch.cat((torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                                   requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        x_mov.unsqueeze(1).unsqueeze(1)), 2),
             torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                                   requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        y_mov.unsqueeze(1).unsqueeze(1)), 2),
             torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                    requires_grad=True, device=device),
                        torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                                   requires_grad=True, device=device),
                        z_mov.unsqueeze(1).unsqueeze(1)), 2),
             torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                          device=device).repeat(BATCH_SIZE, 1, 1)), 1)
        rot_mat = rotation_z.bmm(rotation_y).bmm(rotation_x)
        transform_mat3x4 = torch.bmm(rot_mat, trans_mat)[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)
        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                 src, det, pix_spacing, step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1, 2)).view(BATCH_SIZE, H, W, -1, 3)
        x_3d = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        drr = torch.sum(x_3d, dim=-1)
        return drr, rtvec, pred


class RTPInet_v2(nn.Module):
    '''
    Pose parameter Initialization of Rigid Transformation network
    the input of this network is  a  CT and a fluoroscopy image
    And this network is used to predict the initial parameter of a rigid transformation
    three rotation parameter(rx,ry,tz) and three translation parameter(tx,ty,tz)
    In this version, we take the STN we used in training phase out of the module.
    '''

    def __init__(self):
        super(RTPInet_v2, self).__init__()
        self.block3d1 = Conv3dBlock(1, 64, stride=1)
        self.block3d2 = Conv3dBlock(64, 32, stride=1)
        self.block3d3 = Conv3dBlock(32, 32, stride=1)
        self.fl3d = nn.Flatten(2, 4)
        self.block2d1 = Conv2dBlock(1, 64, stride=1)
        self.block2d2 = Conv2dBlock(64, 32, stride=1)
        self.localization = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(32, 32),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(32, 16, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(16, 16),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(16, 8, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(8, 8),
                                          nn.MaxPool2d(2),
                                          )
        self.fl2d = nn.Flatten(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 6)
        initial_transformation_params = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
        initial_transformation_params = torch.tensor(initial_transformation_params, device="cuda", requires_grad=True)
        self.fc2.bias = Parameter(initial_transformation_params)

    def forward(self, x, y, norm_factor, flag=-1):
        rx = self.block3d1(x)
        rx = self.block3d2(rx)
        rx = self.block3d3(rx)
        rx = self.fl3d(rx)
        rx = rx.view(-1, 32, 64, 64)
        ry = self.block2d1(y)
        ry = self.block2d2(ry)
        out = torch.cat([rx, ry], dim=1)
        out = self.localization(out)
        out = self.fl2d(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        # max_value = 1.0
        # Threshold = 0.0
        out = F.relu6(out)
        out = out / 6
        # print('out:', out.cpu().detach().numpy().squeeze())
        angle_x_min, angle_x_max = 7 / 18 * PI, 11 / 18 * PI
        rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
        angle_yz_min, angle_yz_max = -1 / 18 * PI, 1 / 18 * PI
        ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
        rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
        x_mov_min = x_mov_max = np.random.uniform(-1, 1, (1))[0] * 30 + 700
        tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
        y_mov_min, y_mov_max = -30, 30
        ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
        z_mov_min, z_mov_max = -30, 30
        tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        pred = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        pred[:, 0] = rx * 180 / PI
        pred[:, 1] = ry * 180 / PI
        pred[:, 2] = rz * 180 / PI
        pred[:, 3] = tx * norm_factor
        pred[:, 4] = ty * norm_factor
        pred[:, 5] = tz * norm_factor
        return pred


class RTPInet_v3(nn.Module):
    '''
    Pose parameter Initialization of Rigid Transformation network (version 3)
    the input of this network is  a  CT and a fluoroscopy image
    And this network is used to predict the initial parameter of a rigid transformation
    three rotation parameter(rx,ry,tz) and three translation parameter(tx,ty,tz)
    The average running time of this module is less than 0.1s
    '''

    def __init__(self):
        super(RTPInet_v3, self).__init__()
        self.downsample = nn.Conv3d(1, 1, 2, 2)
        self.conv3d1 = Conv3dBlock(1, 1, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.block3d1 = Conv3dBlock(1, 64, stride=1)
        self.block3d2 = Conv3dBlock(64, 128, stride=1)
        self.conv3d2 = nn.Conv3d(128, 16, kernel_size=1, stride=1)
        self.block2d1 = Conv2dBlock(1, 64, stride=1)
        self.block2d2 = Conv2dBlock(64, 128, stride=1)
        self.conv2d = nn.Conv2d(256, 128, 3, 1, 1)
        self.localization = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(64, 64),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(64, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(32, 32),
                                          nn.MaxPool2d(2),
                                          nn.Conv2d(32, 16, 3, 1, 1),
                                          nn.ReLU(),
                                          SplatBottleneck(16, 16),
                                          nn.MaxPool2d(2),
                                          )
        self.fl2d = nn.Flatten(1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 6)
        initial_transformation_params = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
        initial_transformation_params = torch.tensor(initial_transformation_params, device="cuda",
                                                     requires_grad=True)
        self.fc3.bias = Parameter(initial_transformation_params)

    def forward(self, x, y, norm_factor, flag=-1):
        # print(flag)
        # rx = F.relu(self.downsample(x))
        rx = self.conv3d1(x)
        rx = self.pool1(rx)
        rx = self.block3d1(rx)
        rx = self.block3d2(rx)
        rx = self.conv3d2(rx)
        rx = rx.view(-1, 128, 64, 64)
        ry = self.block2d1(y)
        ry = self.block2d2(ry)
        out = torch.cat([rx, ry], dim=1)
        out = self.conv2d(out)
        out = self.localization(out)
        # print(out.shape)
        out = self.fl2d(out)
        # print(out.shape)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # max_value = 1.0
        # Threshold = 0.0
        out = F.relu6(out)
        out = out / 6
        # print('out:', out.cpu().detach().numpy().squeeze())
        angle_x_min, angle_x_max = 7 / 18 * PI, 11 / 18 * PI
        rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
        angle_yz_min, angle_yz_max = -1 / 18 * PI, 1 / 18 * PI
        ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
        rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
        x_mov_min, x_mov_max = 600, 750
        tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
        y_mov_min, y_mov_max = -30, 30
        ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
        z_mov_min, z_mov_max = -30, 30
        tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        pred = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        pred[:, 0] = rx * 180 / PI
        pred[:, 1] = ry * 180 / PI
        pred[:, 2] = rz * 180 / PI
        pred[:, 3] = tx * norm_factor
        pred[:, 4] = ty * norm_factor
        pred[:, 5] = tz * norm_factor
        return pred
