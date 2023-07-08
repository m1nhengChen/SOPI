import numpy as np
import torch
import torch.nn as nn
import torchgeometry as tgm
import SimpleITK as sitk
import cv2
import math
from sklearn.metrics.cluster import mutual_info_score as MI

flag = 4
proj_x = 256
proj_y = 256
PI = math.pi
criterion = nn.MSELoss()


def hounsfield2linearatten(vol):
    vol = vol.astype(float)
    mu_water_ = 0.02683 * 1.0
    mu_air_ = 0.02485 * 0.0001
    hu_lower_ = -1000
    hu_scale_ = (mu_water_ - mu_air_) * 0.001
    mu_lower_ = (hu_lower_ * hu_scale_) + mu_water_
    for x in np.nditer(vol, op_flags=['readwrite']):
        x[...] = np.maximum((x * hu_scale_) + mu_water_ - mu_lower_, 0.0)

    return vol


# Convert CT HU value to attenuation line integral
def conv_hu_to_density(vol):
    vol = vol.astype(float)
    mu_water_ = 0.02683 * 1.0
    mu_air_ = 0.02485 * 0.0001
    hu_lower_ = -130
    hu_scale_ = (mu_water_ - mu_air_) * 0.001
    mu_lower_ = (hu_lower_ * hu_scale_) + mu_water_
    densities = np.maximum((vol * hu_scale_) + mu_water_ - mu_lower_, 0)
    return densities


def tensor_exp2torch(T, BATCH_SIZE, device):
    T = np.expand_dims(T, axis=0)
    T = np.expand_dims(T, axis=0)
    T = np.repeat(T, BATCH_SIZE, axis=0)

    T = torch.tensor(T, dtype=torch.float, requires_grad=True, device=device)

    return T


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
Defines ProST canonical geometries
input:
    CT_PATH, SEG_PATH: file path of CT and segmentation
    vol_spacing: needs to be calculated offline
    ISFlip: True if Z(IS) is flipped
output:
           param: src, det, pix_spacing, step_size, det_size
         _3D_vol: volume used for training DeepNet, we use CT segmentation
          CT_vol: CT
    ray_proj_mov: detector plane variable
       corner_pt: 8 corner points of input volume
     norm_factor: translation normalization factor
'''


def input_param(CT_PATH, BATCH_SIZE, flag=1, proj_x=1024, ISFlip=False, device='cuda'):
    ct_vol = sitk.ReadImage(CT_PATH)
    vol_spacing = ct_vol.GetSpacing()[1]
    ct_vol = sitk.GetArrayFromImage(ct_vol)
    ct_vol = ct_vol.transpose((2, 1, 0))
    N = ct_vol.shape[0]
    pixel_id_detect = 0.19959  # 老数据是0.304688 0.19959
    src_det = 5069.9 * pixel_id_detect
    iso_center = src_det - N / 2 * pixel_id_detect
    det_size = proj_x
    pix_spacing = pixel_id_detect * flag  # 0.194*1536 / det_size  pixelId_detect * 140 / det_size
    step_size = 2
    vol_size = 512 / flag

    norm_factor = (vol_size * vol_spacing / 2)
    src = (src_det - iso_center) / norm_factor
    det = -iso_center / norm_factor
    pix_spacing = pix_spacing / norm_factor
    step_size = step_size / norm_factor
    param = [src, det, pix_spacing, step_size, det_size]

    ct_vol = tensor_exp2torch(ct_vol, BATCH_SIZE, device)
    corner_pt = create_cornerpt(BATCH_SIZE, device)
    ray_proj_mov = np.zeros((det_size, det_size))
    ray_proj_mov = tensor_exp2torch(ray_proj_mov, BATCH_SIZE, device)
    return param, det_size, ct_vol, ray_proj_mov, corner_pt, norm_factor


def input_param_xray(XRAY_PATH, CT_PATH, BATCH_SIZE, flag=1, proj_x=1024, ISFlip=False, device='cuda'):
    xray_det = sitk.ReadImage(XRAY_PATH)
    xray_det = sitk.GetArrayFromImage(xray_det).reshape(proj_x, proj_x)
    ct_vol = sitk.ReadImage(CT_PATH)
    vol_spacing = ct_vol.GetSpacing()[1]
    ct_vol = sitk.GetArrayFromImage(ct_vol)
    ct_vol = ct_vol.transpose((2, 1, 0))
    N = ct_vol.shape[0]
    pixel_id_detect = 0.19959  # 老数据是0.304688 0.19959
    src_det = 5069.9 * pixel_id_detect
    iso_center = src_det - N / 2 * pixel_id_detect
    det_size = proj_x
    pix_spacing = pixel_id_detect * flag  # 0.194*1536 / det_size  pixelId_detect * 140 / det_size
    step_size = 2
    vol_size = 512 / flag

    norm_factor = (vol_size * vol_spacing / 2)
    src = (src_det - iso_center) / norm_factor
    det = -iso_center / norm_factor
    pix_spacing = pix_spacing / norm_factor
    step_size = step_size / norm_factor
    param = [src, det, pix_spacing, step_size, det_size]

    xray_det = tensor_exp2torch(xray_det, BATCH_SIZE, device)
    ct_vol = tensor_exp2torch(ct_vol, BATCH_SIZE, device)
    corner_pt = create_cornerpt(BATCH_SIZE, device)
    ray_proj_mov = np.zeros((det_size, det_size))
    ray_proj_mov = tensor_exp2torch(ray_proj_mov, BATCH_SIZE, device)

    return param, det_size, xray_det, ct_vol, ray_proj_mov, corner_pt, norm_factor


def init_rtvec_train(BATCH_SIZE, device, norm_factor):
    rz_tar, rx_tar, ry_tar, tx_tar, ty_tar, tz_tar = \
        np.array([90, 0, 0, 900, 0, 0], dtype=np.float32)
    rz_tar_sca, rx_tar_sca, ry_tar_sca, tx_tar_sca, ty_tar_sca, tz_tar_sca = \
        np.array([20, 20, 20, 100, 30, 15], dtype=np.float32)
    rz_ini_sca, rx_ini_sca, ry_ini_sca, tx_ini_sca, ty_ini_sca, tz_ini_sca = \
        np.array([20, 10, 10, 100, 30, 15], dtype=np.float32)
    rz_lateral_tar, rx_lateral_tar, ry_lateral_tar, tx_lateral_tar, ty_lateral_tar, tz_lateral_tar \
        = np.array([180, 0, 0, 800, 0, 0], dtype=np.float32)
    rz_lateral_sca, rx_lateral_sca, ry_lateral_sca, tx_lateral_sca, ty_lateral_sca, tz_lateral_sca = \
        np.array([10, 2, 2, 50, 10, 10], dtype=np.float32)
    target = np.repeat([[rz_tar, ry_tar, rx_tar, -tz_tar, -ty_tar, tx_tar]],
                       BATCH_SIZE, 0)
    target_scale = np.repeat([[rz_tar_sca, ry_tar_sca, rx_tar_sca, -tz_tar_sca, -ty_tar_sca, tx_tar_sca]],
                             BATCH_SIZE, 0)
    initial_scale = np.repeat([[rz_ini_sca, ry_ini_sca, rx_ini_sca, -tz_ini_sca, -ty_ini_sca, tx_ini_sca]],
                              BATCH_SIZE, 0)
    lateral = np.repeat(
        [[rz_lateral_tar, ry_lateral_tar, rx_lateral_tar, -tz_lateral_tar, -ty_lateral_tar, tx_lateral_tar]],
        BATCH_SIZE, 0)
    lateral_scale = np.repeat(
        [[rz_lateral_sca, ry_lateral_sca, rx_lateral_sca, -tz_lateral_sca, -ty_lateral_sca, tx_lateral_sca]],
        BATCH_SIZE, 0)
    # Uniform Distribution/Normal distribution
    target_param = np.random.normal(0, 0.3, (BATCH_SIZE, 6)) * target_scale + target
    initial_param = np.random.normal(0, 0.3, (BATCH_SIZE, 6)) * initial_scale + target_param
    lateral_param = np.random.normal(0, 0.3, (BATCH_SIZE, 6)) * lateral_scale + lateral
    target_param[:, :3] = target_param[:, :3] / 180 * PI
    target_param[:, 3:] = target_param[:, 3:] / norm_factor
    initial_param[:, :3] = initial_param[:, :3] / 180 * PI
    initial_param[:, 3:] = initial_param[:, 3:] / norm_factor
    lateral_param[:, :3] = lateral_param[:, :3] / 180 * PI
    lateral_param[:, 3:] = lateral_param[:, 3:] / norm_factor
    target_torch = torch.tensor(target_param, dtype=torch.float, requires_grad=True, device=device)
    initial_torch = torch.tensor(initial_param, dtype=torch.float, requires_grad=True, device=device)
    lateral_torch = torch.tensor(lateral_param, dtype=torch.float, requires_grad=True, device=device)
    transform_mat3x4_gt = set_matrix(BATCH_SIZE, device, target_torch)
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, initial_torch)

    return transform_mat3x4, transform_mat3x4_gt, initial_torch, target_torch, lateral_torch


def init_rtvec_test(rtvec_gt_param, BATCH_SIZE, device, norm_factor):
    rz_tar, rx_tar, ry_tar, tx_tar, ty_tar, tz_tar = rtvec_gt_param
    target = np.repeat([[rz_tar, ry_tar, rx_tar, -tz_tar, -ty_tar, tx_tar]],
                       BATCH_SIZE, 0)
    rz_ini_sca, rx_ini_sca, ry_ini_sca, tx_ini_sca, ty_ini_sca, tz_ini_sca = \
        np.array([20, 10, 10, 100, 30, 15], dtype=np.float32)
    init_sca = np.repeat([[rz_ini_sca, ry_ini_sca, rx_ini_sca, -tz_ini_sca, -ty_ini_sca, tx_ini_sca]],
                         BATCH_SIZE, 0)
    init = np.random.normal(0, 0.3, (BATCH_SIZE, 6)) * init_sca + target
    target[:, :3] = target[:, :3] / 180 * PI
    target[:, 3:] = target[:, 3:] / norm_factor
    init[:, :3] = init[:, :3] / 180 * PI
    init[:, 3:] = init[:, 3:] / norm_factor
    rtvec_gt = torch.tensor(target, dtype=torch.float, requires_grad=True, device=device)
    rtvec = torch.tensor(init, dtype=torch.float, requires_grad=True, device=device)
    transform_mat3x4_gt = set_matrix(BATCH_SIZE, device, rtvec_gt)
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
    return transform_mat3x4, transform_mat3x4_gt, rtvec, rtvec_gt


def create_cornerpt(BATCH_SIZE, device):
    corner_pt = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    corner_pt = torch.tensor(corner_pt.astype(float), requires_grad=False).type(torch.FloatTensor)
    corner_pt = corner_pt.unsqueeze(0).to(device)
    corner_pt = corner_pt.repeat(BATCH_SIZE, 1, 1)

    return corner_pt


def _repeat(x, n_repeats):
    with torch.no_grad():
        rep = torch.ones((1, n_repeats), dtype=torch.float32).cuda()

    return torch.matmul(x.view(-1, 1), rep).view(-1)


def _bilinear_interpolate_no_torch_5D(vol, grid):
    # Assume CT to be Nx1xDxHxW
    num_batch, channels, depth, height, width = vol.shape
    vol = vol.permute(0, 2, 3, 4, 1)
    _, out_depth, out_height, out_width, _ = grid.shape
    x = width * (grid[:, :, :, :, 0] * 0.5 + 0.5)
    y = height * (grid[:, :, :, :, 1] * 0.5 + 0.5)
    z = depth * (grid[:, :, :, :, 2] * 0.5 + 0.5)

    x = x.view(-1)
    y = y.view(-1)
    z = z.view(-1)

    ind = ~((x >= 0) * (x <= width) * (y >= 0) * (y <= height) * (z >= 0) * (z <= depth))
    # do sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)
    z1 = torch.clamp(z1, 0, depth - 1)

    dim3 = float(width)
    dim2 = float(width * height)
    dim1 = float(depth * width * height)
    dim1_out = float(out_depth * out_width * out_height)

    base = _repeat(torch.arange(start=0, end=num_batch, dtype=torch.float32).cuda() * dim1, np.int32(dim1_out))
    idx_a = base.long() + (z0 * dim2).long() + (y0 * dim3).long() + x0.long()
    idx_b = base.long() + (z0 * dim2).long() + (y0 * dim3).long() + x1.long()
    idx_c = base.long() + (z0 * dim2).long() + (y1 * dim3).long() + x0.long()
    idx_d = base.long() + (z0 * dim2).long() + (y1 * dim3).long() + x1.long()
    idx_e = base.long() + (z1 * dim2).long() + (y0 * dim3).long() + x0.long()
    idx_f = base.long() + (z1 * dim2).long() + (y0 * dim3).long() + x1.long()
    idx_g = base.long() + (z1 * dim2).long() + (y1 * dim3).long() + x0.long()
    idx_h = base.long() + (z1 * dim2).long() + (y1 * dim3).long() + x1.long()

    # use indices to lookup pixels in the flat image and keep channels dim
    im_flat = vol.contiguous().view(-1, channels)
    Ia = im_flat[idx_a].view(-1, channels)
    Ib = im_flat[idx_b].view(-1, channels)
    Ic = im_flat[idx_c].view(-1, channels)
    Id = im_flat[idx_d].view(-1, channels)
    Ie = im_flat[idx_e].view(-1, channels)
    If = im_flat[idx_f].view(-1, channels)
    Ig = im_flat[idx_g].view(-1, channels)
    Ih = im_flat[idx_h].view(-1, channels)

    wa = torch.mul(torch.mul(x1 - x, y1 - y), z1 - z).view(-1, 1)
    wb = torch.mul(torch.mul(x - x0, y1 - y), z1 - z).view(-1, 1)
    wc = torch.mul(torch.mul(x1 - x, y - y0), z1 - z).view(-1, 1)
    wd = torch.mul(torch.mul(x - x0, y - y0), z1 - z).view(-1, 1)
    we = torch.mul(torch.mul(x1 - x, y1 - y), z - z0).view(-1, 1)
    wf = torch.mul(torch.mul(x - x0, y1 - y), z - z0).view(-1, 1)
    wg = torch.mul(torch.mul(x1 - x, y - y0), z - z0).view(-1, 1)
    wh = torch.mul(torch.mul(x - x0, y - y0), z - z0).view(-1, 1)

    interpolated_vol = torch.mul(wa, Ia) + torch.mul(wb, Ib) + torch.mul(wc, Ic) + torch.mul(wd, Id) \
                       + torch.mul(we, Ie) + torch.mul(wf, If) + torch.mul(wg, Ig) + torch.mul(wh, Ih)
    interpolated_vol[ind] = 0.0
    interpolated_vol = interpolated_vol.view(num_batch, out_depth, out_height, out_width, channels)
    interpolated_vol = interpolated_vol.permute(0, 4, 1, 2, 3)

    return interpolated_vol

def set_param(device, proj_parameters):
    proj_parameters = np.array(
        [proj_parameters[0], proj_parameters[2], proj_parameters[1], -proj_parameters[4], -proj_parameters[3],
         proj_parameters[6]], dtype=np.float32)
    pixel_id_voxel = 0.779297  # 新数据是0.808594
    pixel_id_detect = 0.19959  # 老数据是0.304688 0.19959
    offset = 0.01  # 加个偏移值防止cuda计算发生除0
    cs = proj_x / 2  # 竖直改变，取proj_x 中间值
    ls = proj_y / 2  # 水平改变，取proj_y 中间值
    cs = cs + offset
    ls = ls + offset
    sp = pixel_id_detect * flag  # 板像素分辨率，默认体数据分辨率为1
    d = 5069.9 * pixel_id_detect
    # proj_mat = np.array([[cs / d, 1 / sp, 0, 0],
    #                      [ls / d, 0, 1 / sp, 0],
    #                      [1 / d, 0, 0, 0], ], dtype=np.float32)
    # print(p.shape, proj_parameters_torch[:, 3:].T.shape)
    angle_x = proj_parameters[0][0]
    angle_y = proj_parameters[0][1]
    angle_z = proj_parameters[0][2]
    x_mov = proj_parameters[0][3]
    y_mov = proj_parameters[0][4]
    z_mov = proj_parameters[0][5]
    rotation_x = np.array([[1, 0, 0, 0],
                           [0, cos_d(angle_x), -sin_d(angle_x), 0],
                           [0, sin_d(angle_x), cos_d(angle_x), 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    rotation_y = np.array([[cos_d(angle_y), 0, sin_d(angle_y), 0],
                           [0, 1, 0, 0],
                           [-sin_d(angle_y), 0, cos_d(angle_y), 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    rotation_z = np.array([[cos_d(angle_z), -sin_d(angle_z), 0, 0],
                           [sin_d(angle_z), cos_d(angle_z), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    trans_mat = np.array([[1, 0, 0, x_mov],
                          [0, 1, 0, y_mov],
                          [0, 0, 1, z_mov],
                          [0, 0, 0, 1]], dtype=np.float32)
    rot_mat = rotation_z.dot(rotation_y).dot(rotation_x)
    transform_mat3x4 = np.dot(rot_mat, trans_mat)[:3, :]
    # transform_mat3x4 = np.dot(trans_mat, rot_mat)[:3, :]
    transform_mat3x4_gt = torch.tensor(transform_mat3x4, dtype=torch.float, requires_grad=True, device=device)
    transform_mat3x4_gt = torch.unsqueeze(transform_mat3x4_gt, dim=0)
    # Normalization and conversion to transformation matrix
    proj_parameters[:, :3] = proj_parameters[:, :3] * PI / 180
    # print(manual_rtvec)
    # print(norm_factor)
    proj_parameters[:, 3:] = setTrans(proj_parameters[0][0], proj_parameters[0][1],
                                      proj_parameters[0][2], proj_parameters[0][3],
                                      proj_parameters[0][4], proj_parameters[0][5])
    # transform_mat3x4_gt, rtvec = init_rtvec(device, manual_test=True,
    #                                         manual_rtvec=manual_rtvec)
    rtvec_gt = proj_parameters.copy()
    return transform_mat3x4_gt, rtvec_gt


def cos_d(angle):
    if angle > 2 * PI:
        return math.cos(angle % (2 * PI))
    else:
        return math.cos(angle)


def sin_d(angle):
    if angle > 2 * PI:
        return math.sin(angle % (2 * PI))
    else:
        return math.sin(angle)


def setTrans(rx, ry, rz, tx, ty, tz):
    zmat = np.mat([[cos_d(rz), -sin_d(rz), 0],
                   [sin_d(rz), cos_d(rz), 0],
                   [0, 0, 1]])
    cosy = cos_d(ry)
    siny = sin_d(ry)
    ymat = np.mat([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])

    cosx = cos_d(rx)
    sinx = sin_d(rx)
    xmat = np.mat([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])

    rotMat = np.dot(xmat, np.dot(ymat, zmat))
    # print(rotMat)
    tmat = np.mat([[tx], [ty], [tz]])
    # print(tmat)
    result = np.dot(rotMat, tmat)
    # result = np.dot(p,np.dot(rotMat, tmat))
    result = np.transpose(result)
    return result


def set_matrix(BATCH_SIZE, device, proj_parameters):
    radian_x = proj_parameters[:, 0]
    radian_y = proj_parameters[:, 1]
    radian_z = proj_parameters[:, 2]
    x_mov = proj_parameters[:, 3]
    y_mov = proj_parameters[:, 4]
    z_mov = proj_parameters[:, 5]
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
    return transform_mat3x4

def ORB(X, Y):
    '''
    Oriented FAST and Rotated BRIEF
    '''
    orb = cv2.ORB_create()
    kpX, desX = orb.detectAndCompute(X, None)
    kpY, desY = orb.detectAndCompute(Y, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desX, desY)
    return matches


def ORB_distance(X, Y, npoint=20):
    '''
    the mean distance of the matched points
    npoint->the number of matched point-pairs
    '''
    matches = ORB(X, Y)
    orb_distances = []
    for i in matches:
        orb_distances.append(i.distance)
    orb_distance = np.array(orb_distances)
    orb_distance.sort()
    orb_distance = orb_distance[:npoint]
    orb_distance_mean = np.mean(orb_distance)
    return orb_distance_mean


def SURF(X, Y):
    '''Speeded-Up Robust Features (SURF)'''
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(X, None)
    kp2, des2 = surf.detectAndCompute(Y, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    return matches


def SURF_distance(X, Y, npoint=20):
    '''
        the mean distance of the matched points
        npoint->the number of matched point-pairs
    '''
    matches = SURF(X, Y)
    surf_distances = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            surf_distances.append(m.distance)
    surf_distance = np.array(surf_distances)
    surf_distance.sort()
    surf_distance = surf_distance[:npoint]
    surf_distance_mean = np.mean(surf_distance)
    return surf_distance_mean
    
# Convert pose parameters to rotation-translation vector
def pose2rtvec(pose, device, norm_factor):
    rtvec = torch.zeros_like(pose, dtype=torch.float, requires_grad=False, device=device)
    rtvec[:, 0] = pose[:, 0]
    rtvec[:, 1] = pose[:, 2]
    rtvec[:, 2] = pose[:, 1]
    rtvec[:, 3] = - pose[:, 5]
    rtvec[:, 4] = - pose[:, 4]
    rtvec[:, 5] = pose[:, 3]
    rtvec[:, :3] = rtvec[:, :3] / 180 * PI
    rtvec[:, 3:] = rtvec[:, 3:] / norm_factor
    return rtvec


# Convert rotation-translation vector to pose parameters
def rtvec2pose(rtvec, norm_factor, device):
    pose = torch.zeros_like(rtvec, dtype=torch.float, requires_grad=False, device=device)
    pose[:, 0] = rtvec[:, 0]
    pose[:, 1] = rtvec[:, 2]
    pose[:, 2] = rtvec[:, 1]
    pose[:, 3] = rtvec[:, 5]
    pose[:, 4] = - rtvec[:, 4]
    pose[:, 5] = - rtvec[:, 3]
    pose[:, :3] = pose[:, :3] * 180 / PI
    pose[:, 3:] = pose[:, 3:] * norm_factor
    return pose


# Convert pose parameters to transform matrix
def pose2mat(pose, BATCH_SIZE, device, norm_factor):
    rtvec = pose2rtvec(pose, device, norm_factor)
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
    return transform_mat3x4


# Seed everything for random steps
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Early stop for training
def early_stop(val_loss, val_loss_list, EARLY_STOP_COUNTER, EARLY_STOP_THRESHOLD):
    val_loss_min = min(val_loss_list)
    if val_loss > val_loss_min:
        EARLY_STOP_COUNTER += 1
    if EARLY_STOP_COUNTER >= EARLY_STOP_THRESHOLD:
        return 1
    return 0
