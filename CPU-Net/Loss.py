import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import SimpleITK as sitk
import math


# def normalize(image):
#     # image为sitk的图像
#     # 返回的是sitk的图像，范围规格化到0-255
#     image_np = sitk.GetArrayViewFromImage(image)
#     oriMin = float(np.min(image_np))
#     oriMax = float(np.max(image_np))
#     oriRange = oriMax - oriMin
#
#     desiredMin = 0
#     desiredMax = 255
#     desiredRange = desiredMax - desiredMin
#
#     image_255_np = desiredRange * (image_np - oriMin) / oriRange + desiredMin
#     image_255 = sitk.Cast(sitk.GetImageFromArray(image_255_np), sitk.sitkFloat32)
#
#     return image_255


class NCC(nn.Module):
    def __init__(self):
        super(NCC, self).__init__()

    def forward(self, image_fixed, image_moving):
        # NCC方法
        imageA = (image_fixed - image_fixed.min()) / (image_fixed.max() - image_fixed.min()) * 255
        imageB = (image_moving - image_moving.min()) / (image_moving.max() - image_moving.min()) * 255

        #     imageA=imageA[0,:,:]
        #     imageB=imageB[0,:,:]
        meanA = np.sum(imageA) / float(imageA.shape[0] * imageA.shape[1])
        meanB = np.sum(imageB) / float(imageA.shape[0] * imageA.shape[1])
        D_A = math.sqrt(np.sum((imageA - meanA) ** 2))
        D_B = math.sqrt(np.sum((imageB - meanB) ** 2))
        ncc = np.sum((imageA - meanA) * (imageB - meanB) / (D_A * D_B)) * 1000

        #     print("当前误差",err)
        return ncc
