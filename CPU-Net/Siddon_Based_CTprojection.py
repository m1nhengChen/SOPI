import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import time
T1 = time.time()
block_size = 1024
dx = 0.
dy = 0.
dz = 0.
pixel_d = 0.8
pixel_dz = 0.8
image_x = 128
image_y = 128
image_z = 250
proj_x = 128
proj_y = 128


def cos_d(angle):
    return math.cos(angle / 180 * math.pi)


def sin_d(angle):
    return math.sin(angle / 180 * math.pi)


def set_matrix(proj_parameters):
    angle_z = proj_parameters[0]
    angle_x = proj_parameters[1]
    angle_y = proj_parameters[2]
    x_mov = proj_parameters[3]
    y_mov = proj_parameters[4]
    z_mov = proj_parameters[5]
    pixel_id_detect = 0.19959  # 老数据是0.304688
    offset = 0.01  # 加个偏移值防止cuda计算发生除0
    cs = proj_x / 2  # 竖直改变，取proj_x 中间值
    ls = proj_y / 2  # 水平改变，取proj_y 中间值
    cs = cs + offset
    ls = ls + offset
    sp = pixel_id_detect  # 板像素分辨率，默认体数据分辨率为1
    d = 1000
    xs = -550
    p = torch.tensor([[cs / d, 1 / sp, 0],
                      [ls / d, 0, 1 / sp],
                      [1 / d, 0, 0]
                      ])
    rotation_x = torch.Tensor([[1, 0, 0, 0],
                               [0, cos_d(90 + angle_x), -sin_d(90 + angle_x), 0],
                               [0, sin_d(90 + angle_x), cos_d(90 + angle_x), 0],
                               [0, 0, 0, 1]])
    rotation_y = torch.tensor([[cos_d(angle_y), 0, sin_d(angle_y), 0],
                               [0, 1, 0, 0],
                               [-sin_d(angle_y), 0, cos_d(angle_y), 0],
                               [0, 0, 0, 1]])
    rotation_z = torch.tensor([[cos_d(angle_z), -sin_d(angle_z), 0, 0],
                               [sin_d(angle_z), cos_d(angle_z), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])
    transform_1 = torch.tensor([[1, 0, 0, -xs],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]])
    transform_2 = torch.tensor([[1, 0, 0, x_mov],
                                [0, 1, 0, y_mov],
                                [0, 0, 1, z_mov],
                                [0, 0, 0, 1]])
    result = np.dot(p, np.dot(transform_1, np.dot(transform_2, np.dot(rotation_x, np.dot(rotation_y, rotation_z)))))
    result = torch.tensor(result, requires_grad=True, dtype=torch.float32).reshape(12)
    return result


# @cuda.jit()
# def cuda_image_pre_operate(image, result):
#     j = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
#     if image[j] < 1:
#         result[j] = 0
#     else:
#         result[j] = image[j]
#
#
# def image_pre_operate(image_flattened):
#     image_flattened_device = cuda.to_device(image_flattened)
#     result_device = cuda.device_array(512 * 512 * 906)
#     cuda_image_pre_operate[231936, 1024](image_flattened_device, result_device)
#     image_3d = result_device.copy_to_host()
#     return image_3d


def cpu_project(image_3d, image_2d, transform_parameter, j):
    # transform_parameter：(float)numpy[12]
    if j >= proj_x * proj_y:
        return
    u = j // proj_y
    v = j % proj_y
    # if (v<30 or u<2): return
    flag = 0
    # if(u<0 or u>511 or v<0 or v>511): return
    point = torch.zeros((3, 4), dtype=torch.float32)
    p_xyz = torch.zeros(6, dtype=torch.float32)
    alpha_x = torch.zeros(image_x + 1, dtype=torch.float32)
    alpha_y = torch.zeros(image_y + 1, dtype=torch.float32)
    alpha_z = torch.zeros(image_z + 1, dtype=torch.float32)
    alpha_xy = torch.zeros(image_x + image_y + 2, dtype=torch.float32)
    alpha = torch.zeros(image_x + image_y + image_z + 3, dtype=torch.float32)
    # 寻找每个投影点对应的三维体数据的起始点和终点
    # (zb,yb,zb)与(xe,ye,ze)是3D体数据在三维空间的最高值点与最低值点，确定其在空间中的位置，设置体数据的几何中心在坐标原点
    py = 0.
    xb = (0 - image_x / 2) * pixel_d + dx
    yb = (0 - image_y / 2) * pixel_d + dy
    zb = (0 - image_z / 2) * pixel_dz + dz - py
    xe = (image_x - 1 - image_x / 2) * pixel_d + dx
    ye = (image_y - 1 - image_y / 2) * pixel_d + dy
    ze = (image_z - 1 - image_z / 2) * pixel_dz + dz - py
    # 固定z底层
    t1 = transform_parameter[0] - u * transform_parameter[8]
    t2 = transform_parameter[1] - u * transform_parameter[9]
    t3 = transform_parameter[4] - v * transform_parameter[8]
    t4 = transform_parameter[5] - v * transform_parameter[9]
    a = (transform_parameter[10] * zb + transform_parameter[11]) * u - (
            transform_parameter[2] * zb + transform_parameter[3])
    b = (transform_parameter[10] * zb + transform_parameter[11]) * v - (
            transform_parameter[6] * zb + transform_parameter[7])
    t1.retain_grad()
    t2.retain_grad()
    t3.retain_grad()
    t4.retain_grad()
    a.retain_grad()
    b.retain_grad()
    '''// 检查除0
    / * if (t1 * t4 - t2 * t3 < 0.001 & & t1 * t4 - t2 * t3 > -0.001)
    {
        proj_m[j] = 0;
    }
    else {
        proj_m[j] = 1;
    }
    return; * /'''
    point[0][0] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[1][0] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    point.retain_grad()
    p_xyz[0] = point[0][0]
    p_xyz.retain_grad()
    p_xyz[0] = 0
    if xb <= point[0][0] <= xe and yb <= point[1][0] <= ye:
        # print("zb")
        flag += 1
        if flag == 1:
            p_xyz[0] = (point[0][0] - dx) / pixel_d + image_x / 2
            p_xyz[1] = (point[1][0] - dy) / pixel_d + image_y / 2
            p_xyz[2] = (zb - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][0]
        if flag == 2:
            p_xyz[3] = (point[0][0] - dx) / pixel_d + image_x / 2
            p_xyz[4] = (point[1][0] - dy) / pixel_d + image_y / 2
            p_xyz[5] = (zb - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][0]
    # 固定z顶层
    t1 = transform_parameter[0] - u * transform_parameter[8]
    t2 = transform_parameter[1] - u * transform_parameter[9]
    t3 = transform_parameter[4] - v * transform_parameter[8]
    t4 = transform_parameter[5] - v * transform_parameter[9]
    a = (transform_parameter[10] * ze + transform_parameter[11]) * u - (
            transform_parameter[2] * ze + transform_parameter[3])
    b = (transform_parameter[10] * ze + transform_parameter[11]) * v - (
            transform_parameter[6] * ze + transform_parameter[7])
    point[0][1] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[1][1] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    if xb <= point[0][1] <= xe and yb <= point[1][1] <= ye:
        # print("ze")
        flag += 1
        if flag == 1:
            p_xyz[0] = (point[0][1] - dx) / pixel_d + image_x / 2
            p_xyz[1] = (point[1][1] - dy) / pixel_d + image_y / 2
            p_xyz[2] = (ze - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][1]
        if flag == 2:
            p_xyz[3] = (point[0][1] - dx) / pixel_d + image_x / 2
            p_xyz[4] = (point[1][1] - dy) / pixel_d + image_y / 2
            p_xyz[5] = (ze - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][1]
    # 固定y底层
    t1 = transform_parameter[0] - u * transform_parameter[8]
    t2 = transform_parameter[2] - u * transform_parameter[10]
    t3 = transform_parameter[4] - v * transform_parameter[8]
    t4 = transform_parameter[6] - v * transform_parameter[10]
    a = (transform_parameter[9] * yb + transform_parameter[11]) * u - (
            transform_parameter[1] * yb + transform_parameter[3])
    b = (transform_parameter[9] * yb + transform_parameter[11]) * v - (
            transform_parameter[5] * yb + transform_parameter[7])
    point[0][2] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[2][0] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    if xb <= point[0][2] <= xe and zb <= point[2][0] <= ze:
        # print("yb")
        flag += 1
        if flag == 1:
            p_xyz[0] = (point[0][2] - dx) / pixel_d + image_x / 2
            p_xyz[1] = (yb - dy) / pixel_d + image_y / 2
            p_xyz[2] = (point[2][0] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][2]
        if flag == 2:
            p_xyz[3] = (point[0][2] - dx) / pixel_d + image_x / 2
            p_xyz[4] = (yb - dy) / pixel_d + image_y / 2
            p_xyz[5] = (point[2][0] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        # proj_m[j] += point[0][2]

    # 固定y顶层
    t1 = transform_parameter[0] - u * transform_parameter[8]
    t2 = transform_parameter[2] - u * transform_parameter[10]
    t3 = transform_parameter[4] - v * transform_parameter[8]
    t4 = transform_parameter[6] - v * transform_parameter[10]
    a = (transform_parameter[9] * ye + transform_parameter[11]) * u - (
            transform_parameter[1] * ye + transform_parameter[3])
    b = (transform_parameter[9] * ye + transform_parameter[11]) * v - (
            transform_parameter[5] * ye + transform_parameter[7])
    point[0][3] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[2][1] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    if xb <= point[0][3] <= xe and zb <= point[2][1] <= ze:
        # print("ye")
        flag += 1
        if flag == 1:
            p_xyz[0] = (point[0][3] - dx) / pixel_d + image_x / 2
            p_xyz[1] = (ye - dy) / pixel_d + image_y / 2
            p_xyz[2] = (point[2][1] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        if flag == 2:
            p_xyz[3] = (point[0][3] - dx) / pixel_d + image_x / 2
            p_xyz[4] = (ye - dy) / pixel_d + image_y / 2
            p_xyz[5] = (point[2][1] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
    # 固定x底层
    t1 = transform_parameter[1] - u * transform_parameter[9]
    t2 = transform_parameter[2] - u * transform_parameter[10]
    t3 = transform_parameter[5] - v * transform_parameter[9]
    t4 = transform_parameter[6] - v * transform_parameter[10]
    a = (transform_parameter[8] * xb + transform_parameter[11]) * u - (
            transform_parameter[0] * xb + transform_parameter[3])
    b = (transform_parameter[8] * xb + transform_parameter[11]) * v - (
            transform_parameter[4] * xb + transform_parameter[7])
    point[1][2] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[2][2] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    if yb <= point[1][2] <= ye and zb <= point[2][2] <= ze:
        # print("xb")
        flag += 1
        if flag == 1:
            p_xyz[0] = (xb - dx) / pixel_d + image_x / 2
            p_xyz[1] = (point[1][2] - dy) / pixel_d + image_y / 2
            p_xyz[2] = (point[2][2] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        if flag == 2:
            p_xyz[3] = (xb - dx) / pixel_d + image_x / 2
            p_xyz[4] = (point[1][2] - dy) / pixel_d + image_y / 2
            p_xyz[5] = (point[2][2] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
    # 固定x顶层
    t1 = transform_parameter[1] - u * transform_parameter[9]
    t2 = transform_parameter[2] - u * transform_parameter[10]
    t3 = transform_parameter[5] - v * transform_parameter[9]
    t4 = transform_parameter[6] - v * transform_parameter[10]
    a = (transform_parameter[8] * xe + transform_parameter[11]) * u - (
            transform_parameter[0] * xe + transform_parameter[3])
    b = (transform_parameter[8] * xe + transform_parameter[11]) * v - (
            transform_parameter[4] * xe + transform_parameter[7])
    point[1][3] = (t4 * a - t2 * b) / (t1 * t4 - t2 * t3)
    point[2][3] = (t3 * a - t1 * b) / (t2 * t3 - t1 * t4)
    if yb <= point[1][3] <= ye and zb <= point[2][3] <= ze:
        # print("xe")
        flag += 1
        if flag == 1:
            p_xyz[0] = (xe - dx) / pixel_d + image_x / 2
            p_xyz[1] = (point[1][3] - dy) / pixel_d + image_y / 2
            p_xyz[2] = (point[2][3] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
        if flag == 2:
            p_xyz[3] = (xe - dx) / pixel_d + image_x / 2
            p_xyz[4] = (point[1][3] - dy) / pixel_d + image_y / 2
            p_xyz[5] = (point[2][3] - dz + py) / pixel_dz + image_z / 2
            p_xyz.retain_grad()
    # siddon算法
    i_min = math.floor(min(p_xyz[0], p_xyz[3]))
    i_max = math.ceil(max(p_xyz[0], p_xyz[3]))
    j_min = math.floor(min(p_xyz[1], p_xyz[4]))
    j_max = math.ceil(max(p_xyz[1], p_xyz[4]))
    k_min = math.floor(min(p_xyz[2], p_xyz[5]))
    k_max = math.ceil(max(p_xyz[2], p_xyz[5]))
    n = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1)
    # matrix_jkl[0][max_n - 1] = n
    nx = i_max - i_min + 1
    ny = j_max - j_min + 1
    nz = k_max - k_min + 1
    # 三个有序数组，归并排序
    alpha_x[0] = p_xyz[0]
    alpha_x.retain_grad
    alpha_x[0] = 0
    alpha_y[0] = p_xyz[0]
    alpha_y.retain_grad
    alpha_y[0] = 0
    alpha_z[0] = p_xyz[0]
    alpha_z.retain_grad
    alpha_z[0] = 0
    for i in range(i_min, i_max + 1):
        if p_xyz[0] == p_xyz[3]:
            alpha_x[i - i_min] = 1
            break
        if p_xyz[3] > p_xyz[0]:
            alpha_x[i - i_min] = (i - p_xyz[0]) / (p_xyz[3] - p_xyz[0])
        else:
            alpha_x[i - i_min] = ((i_max - i + i_min) - p_xyz[0]) / (p_xyz[3] - p_xyz[0])
        if alpha_x[i - i_min] < 0.:
            alpha_x[i - i_min] = 0.
        elif alpha_x[i - i_min] > 1.:
            alpha_x[i - i_min] = 1.
    for i in range(j_min, j_max + 1):
        if p_xyz[4] == p_xyz[1]:
            alpha_y[i - j_min] = 1
            break
        if p_xyz[4] > p_xyz[1]:
            alpha_y[i - j_min] = (i - p_xyz[1]) / (p_xyz[4] - p_xyz[1])
        else:
            alpha_y[i - j_min] = ((j_max - i + j_min) - p_xyz[1]) / (p_xyz[4] - p_xyz[1])
        if alpha_y[i - j_min] < 0.:
            alpha_y[i - j_min] = 0.
        elif alpha_y[i - j_min] > 1.:
            alpha_y[i - j_min] = 1.
    for i in range(k_min, k_max + 1):
        if p_xyz[5] == p_xyz[2]:
            alpha_z[i - k_min] = 1
            break
        if p_xyz[5] > p_xyz[2]:
            alpha_z[i - k_min] = (i - p_xyz[2]) / (p_xyz[5] - p_xyz[2])
        else:
            alpha_z[i - k_min] = ((k_max - i + k_min) - p_xyz[2]) / (p_xyz[5] - p_xyz[2])
        if alpha_z[i - k_min] < 0.:
            alpha_z[i - k_min] = 0.
        elif alpha_z[i - k_min] > 1.:
            alpha_z[i - k_min] = 1.
    nxy = nx + ny
    index_xy = 0
    index_x = 0
    index_y = 0
    alpha_xy[0] = alpha_x[0]
    alpha_xy.retain_grad()
    alpha_xy[0] = 0
    while index_x < nx and index_y < ny:
        if alpha_x[index_x] <= alpha_y[index_y]:
            alpha_xy[index_xy] = alpha_x[index_x]
            index_xy += 1
            index_x += 1
        else:
            alpha_xy[index_xy] = alpha_y[index_y]
            index_xy += 1
            index_y += 1
    while index_x < nx:
        alpha_xy[index_xy] = alpha_x[index_x]
        index_xy += 1
        index_x += 1
    while index_y < ny:
        alpha_xy[index_xy] = alpha_y[index_y]
        index_xy += 1
        index_y += 1
    #  nxyz=nxy+nz
    index = 0
    index_z = 0
    index_xy = 0
    alpha_z[0] = alpha_xy[0]
    alpha_z.retain_grad()
    alpha_z[0] = 0
    while index_xy < nxy and index_z < nz:
        if alpha_xy[index_xy] <= alpha_z[index_z]:
            alpha[index] = alpha_xy[index_xy]
            index += 1
            index_xy += 1
        else:
            alpha[index] = alpha_z[index_z]
            index += 1
            index_z += 1
    while index_xy < nxy:
        alpha[index] = alpha_xy[index_xy]
        index += 1
        index_xy += 1
    while index_z < nz:
        alpha[index] = alpha_z[index_z]
        index += 1
        index_z += 1
    l = math.sqrt((p_xyz[3] - p_xyz[0]) * (p_xyz[3] - p_xyz[0]) +
                  (p_xyz[4] - p_xyz[1]) * (p_xyz[4] - p_xyz[1]) +
                  (p_xyz[5] - p_xyz[2]) * (p_xyz[5] - p_xyz[2]))
    w = alpha[0]
    w.retain_grad()
    w = 0
    for i in range(1, n):
        a_mid = (alpha[i] + alpha[i - 1]) / 2
        w = (alpha[i] - alpha[i - 1]) * l
        x = math.floor(p_xyz[0] + a_mid * (p_xyz[3] - p_xyz[0]))
        y = math.floor(p_xyz[1] + a_mid * (p_xyz[4] - p_xyz[1]))
        z = math.floor(p_xyz[2] + a_mid * (p_xyz[5] - p_xyz[2]))
        # proj_m[j] +=2.0
        # if(z<0 or z>511)continue
        # print(z * image_xyz[0] * image_xyz[1] + y * image_xyz[0] + x)
        image_2d[j] += image_3d[z * image_x * image_y + y * image_x + x] * w


def project(proj_parameters, index):
    proj_parameters = proj_parameters.detach().numpy()
    image_3d_addr = "./data/final.raw"
    image_flattened = np.fromfile(image_3d_addr, dtype="float32")
    image_3d = image_flattened[4096000 * index:4096000 * (index + 1)]
    image_3d = torch.tensor(image_3d)
    # image_3d = image_pre_operate(image_flattened)
    image_2d = torch.zeros(proj_x * proj_y, dtype=torch.float32)
    transform_parameters = set_matrix(proj_parameters)
    for j in range(proj_x * proj_y):
        cpu_project(image_3d, image_2d, transform_parameters, j)
    return image_2d


if __name__ == "__main__":
    parameter_matrix = torch.tensor((272, 0, 0, 5000, 0, -5))
    image_2d = project(parameter_matrix, 1)
    x = image_2d.detach()
    y = x.numpy()
    y.resize(proj_x, proj_y)
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
    plt.imshow(y, cmap='gray')
    plt.show()
