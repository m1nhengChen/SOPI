import torch
import ProSTGrid
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import _bilinear_interpolate_no_torch_5D, set_matrix
import os
import torchgeometry as tgm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from posevec2mat import pose_vec2mat, inv_pose_vec, raydist_range
import math
from torch.nn.parameter import Parameter

device = torch.device('cuda')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ProST(nn.Module):
    def __init__(self):
        super(ProST, self).__init__()

    def forward(self, x, y, transform_mat3x4, corner_pt, param):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        BATCH_SIZE = transform_mat3x4.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                 src, det, pix_spacing, step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1, 2)).view(BATCH_SIZE, H, W, -1, 3)

        x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x_2d_ad = torch.sum(x_3d_ad, dim=-1)

        return x_2d_ad


class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d"""

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = nn.modules.utils._pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                                 groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                                  groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = nn.DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        assert radix > 0
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplatBottleneck(nn.Module):
    '''Splat attention bottleneck '''
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SplatBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = SplAtConv2d(width, width, 1)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Conv2dBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, dpr=0.2, norm_layer=None):
        super(Conv2dBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv3x3(inplanes, planes, stride, groups, dilation)
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(dpr)
        self.avgPool = nn.AvgPool2d(kernel_size=2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.avgPool(x)

        return x


class Conv3dBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, dpr=0.2, norm_layer=None):
        super(Conv3dBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, groups=groups,
                               padding=dilation)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=stride, groups=groups,
                               padding=dilation)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout3d(dpr)
        self.avgPool = nn.AvgPool3d(kernel_size=2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.avgPool(x)

        return x


class ChannelWiseMultiply(nn.Module):
    def __init__(self, num_channels):
        super(ChannelWiseMultiply, self).__init__()
        self.param = nn.Parameter(torch.FloatTensor(num_channels), requires_grad=True)

    def init_value(self, value):
        with torch.no_grad():
            self.param.data.fill_(value)

    def forward(self, x):
        return torch.mul(self.param.view(1, -1, 1, 1), x)


class RTPInet(nn.Module):
    '''
    Pose parameter Initialization of Rigid Transformation network
    the input of this network is  a  CT and a fluoroscopy image
    And this network is used to predict the initial parameter of a rigid transformation
    three rotation parameter(rx,ry,tz) and three translation parameter(tx,ty,tz)
    '''

    def __init__(self):
        super(RTPInet, self).__init__()
        self.block3d1 = Conv3dBlock(1, 64, stride=1)
        self.block3d2 = Conv3dBlock(64, 32, stride=1)
        self.block3d3 = Conv3dBlock(32, 32, stride=1)
        self.fl3d = nn.Flatten(2, 4)
        self.block2d1 = Conv2dBlock(1, 64, stride=1)
        self.block2d2 = Conv2dBlock(64, 32, stride=1)
        self.fussion_net = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
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
        initial_transformation_params = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]
        initial_transformation_params = torch.tensor(initial_transformation_params, device="cuda", requires_grad=True)
        self.fc2.bias = Parameter(initial_transformation_params)

    def forward(self, x, y, corner_pt, param, norm_factor):
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
        out = self.fussion_net(out)
        out = self.fl2d(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # max_value = 1.0
        # Threshold = 0.0
        out = F.relu6(out)
        out = out / 6
        BATCH_SIZE = out.size()[0]
        H = y.size()[2]
        W = y.size()[3]
        angle_x_min, angle_x_max = 6 / 18 * math.pi, 12 / 18 * math.pi
        rx = out[:, 0] * (angle_x_max - angle_x_min) + angle_x_min
        angle_yz_min, angle_yz_max = -1 / 9 * math.pi, 1 / 9 * math.pi
        ry = out[:, 1] * (angle_yz_max - angle_yz_min) + angle_yz_min
        rz = out[:, 2] * (angle_yz_max - angle_yz_min) + angle_yz_min
        x_mov_min, x_mov_max = 600, 800
        tx = (out[:, 3] * (x_mov_max - x_mov_min) + x_mov_min) / norm_factor
        y_mov_min, y_mov_max = -40, 30
        ty = (out[:, 4] * (y_mov_max - y_mov_min) + y_mov_min) / norm_factor
        z_mov_min, z_mov_max = -15, 20
        tz = (out[:, 5] * (z_mov_max - z_mov_min) + z_mov_min) / norm_factor
        pred = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        rtvec = torch.zeros_like(out, dtype=torch.float, requires_grad=False, device=device)
        pred[:, 0] = rx * 180 / math.pi
        pred[:, 1] = ry * 180 / math.pi
        pred[:, 2] = rz * 180 / math.pi
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



class CovBlock(nn.Module):
    def __init__(self, in_channel, out_channel, use_Leaky_relu=True, bottleneck="Splate-Attention"):
        super(CovBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1)
        if use_Leaky_relu:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.PReLU()
        if bottleneck == "Splate-Attention":
            self.bottleneck = SplatBottleneck(out_channel, out_channel)
        else:
            self.bottleneck = Bottleneck(out_channel, out_channel)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.relu(x)
        x = self.bottleneck(x)

        return x


class Composite_connection_unit(nn.Module):
    '''
    Composite connection unit is used for decreasing dimension of the features
    '''

    def __init__(self, inchannels, outchannels):
        super(Composite_connection_unit, self).__init__()
        self.cov = conv1x1(inchannels, outchannels)
        self.bn = nn.BatchNorm2d(outchannels)

    def forward(self, x):
        out = self.cov(x)
        out = self.bn(out)
        return out


class fine_regnet(nn.Module):
    '''
    This network is used for fine 2d/3d registration
    Joint the task of  global and texture feature extraction
    '''

    def __init__(self, use_Leaky_relu=True, bottleneck="Splate-Attention"):
        super(fine_regnet, self).__init__()
        self.encoderx = RegiEncoder(use_Leaky_relu, bottleneck)
        self.encodery = RegiEncoder(use_Leaky_relu, bottleneck)

    def forward(self, x, y, rtvec, corner_pt, param):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        BATCH_SIZE = rtvec.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat3x4 = set_matrix(BATCH_SIZE, 'cuda', rtvec)
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)
        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                 src, det, pix_spacing, step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1, 2)).view(BATCH_SIZE, H, W, -1, 3)
        x_3d = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x_2d = torch.sum(x_3d, dim=-1)
        x_out_a, x_out_l1, x_out_l2, x_out_l3 = self.encoderx(x_2d)
        y_out_a, y_out_l1, y_out_l2, y_out_l3 = self.encodery(y)

        return x_out_l1, x_out_l2, x_out_l3, y_out_l1, y_out_l2, y_out_l3, x_2d


class stem(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, normchannels1, normchannels2):
        super(stem, self).__init__()
        self.cov1 = nn.Conv2d(inchannels, midchannels, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.LayerNorm(normchannels1)
        self.gelu = nn.GELU()
        self.cov2 = nn.Conv2d(midchannels, outchannels, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.LayerNorm(normchannels2)

    def forward(self, x):
        x = self.cov1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        x = self.cov2(x)
        x = self.norm2(x)
        out = self.gelu(x)
        return out


class transition_block(nn.Module):
    '''
    The transition block in our network is the‘Sequentially Add’ design
    which achieved the best results
    '''

    def __init__(self, c3, c4, c5):
        super(transition_block, self).__init__()
        self.cov1 = conv1x1(c5, c4)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cov2 = conv1x1(c4, c3)

    def forward(self, c3, c4, c5):
        x = self.cov1(c5)
        p5 = self.up(x)
        x = p5 + c4
        x = self.cov2(x)
        p4 = self.up(x)
        p3 = p4 + c3
        return p3, p4, p5


class focal_block(nn.Module):
    '''
    Focal NeXt block
    '''

    def __init__(self, n, r, normchannels1, normchannels2):
        super(focal_block, self).__init__()
        self.cov1 = nn.Conv2d(n, n, kernel_size=7, stride=1, padding=3, dilation=1, groups=n)
        self.norm1 = nn.LayerNorm(normchannels1)
        self.gelu = nn.GELU()
        self.cov2 = nn.Conv2d(n, n, kernel_size=7, stride=1, padding=3 * r, dilation=r, groups=n)
        self.norm2 = nn.LayerNorm(normchannels2)
        self.cov3 = conv1x1(n, 4 * n)
        self.cov4 = conv1x1(4 * n, n)

    def forward(self, x):
        y = self.cov1(x)
        y = self.norm1(y)
        y = self.gelu(y)
        y = x + y
        z = self.cov2(y)
        z = self.norm2(z)
        z = self.gelu(z)
        z = y + z
        z = self.cov3(z)
        z = self.gelu(z)
        z = self.cov4(z)
        out = x + z

        return out


class CFblock(nn.Module):
    '''
    CF-net block architecture,each stage consists of a sub-backbone
    and an extremely lightweight transition block to extract and integrate features of different scales
    '''

    def __init__(self, c3, c4, c5, cn1, cn2, r=2):
        super(CFblock, self).__init__()
        self.block1 = SplatBottleneck(c3, c3)
        self.down1 = nn.Conv2d(c3, c4, kernel_size=3, padding=1, stride=2)
        self.block2 = SplatBottleneck(c4, c4)
        self.down2 = nn.Conv2d(c4, c5, kernel_size=3, padding=1, stride=2)
        self.fblock = focal_block(c5, r, cn1, cn2)
        self.tblock = transition_block(c3, c4, c5)

    def forward(self, x):
        c3 = self.block1(x)
        c4 = self.down1(c3)
        c4 = self.block2(c4)
        c5 = self.down2(c4)
        c5 = self.fblock(c5)
        p3, p4, p5 = self.tblock(c3, c4, c5)

        return p3, p4, p5


class RegiEncoder(nn.Module):
    def __init__(self, use_Leaky_relu=True, bottleneck="Splate-Attention"):
        super(RegiEncoder, self).__init__()
        self.stem = stem(1, 4, 16, 128, 64)
        self.branche1_layer1 = CovBlock(16, 64, use_Leaky_relu, bottleneck)
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)
        self.branche1_layer2 = CFblock(64, 128, 256, 8, 8, 2)
        self.branche1_layer3 = CFblock(64, 128, 256, 8, 8, 2)
        self.branche1_layer4 = CFblock(64, 128, 256, 8, 8, 2)
        self.branche2_layer3 = CFblock(64, 128, 256, 8, 8, 2)
        self.branche2_layer4 = CFblock(64, 128, 256, 8, 8, 2)
        self.cc3 = Composite_connection_unit(64, 64)
        self.cc4 = Composite_connection_unit(64, 64)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.stem(x)
        b1l1 = self.branche1_layer1(x)
        b1l1 = self.down1(b1l1)
        b1l2, _, __ = self.branche1_layer2(b1l1)
        b1l3, _, __ = self.branche1_layer3(b1l2)
        b1l4, _, __ = self.branche1_layer4(b1l3)
        y = b1l2 + self.cc3(b1l3)
        y, _, __ = self.branche2_layer3(y)
        y = y + self.cc4(b1l4)
        p3, p4, p5 = self.branche2_layer4(y)
        branch1_out = b1l4
        branch2_out1 = p3
        branch2_out2 = p4
        branch2_out3 = p5
        return branch1_out, branch2_out1, branch2_out2, branch2_out3
