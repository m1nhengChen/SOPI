import numpy as np
import os
import torch
import SimpleITK as sitk
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
from module import ProST
from util import input_param, init_rtvec_test

SAVE_PATH = "../Data/ct/512"
# matplotlib.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

'''
Downsample and segment a full scan CT dataset from 512x512x512 to 128x128x128(only contain spine) by using corresponding mask
Reference of segmentation method: https://gitlab.inria.fr/spine/vertebrae_segmentation
'''


def mask_calculate(ct, mask):
    ct_array = sitk.GetArrayFromImage(ct)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array_normalized = np.int64(mask_array > 0)
    ct_masked_array = ct_array * mask_array_normalized
    ct_masked = sitk.GetImageFromArray(ct_masked_array)
    ct_masked.SetSpacing(ct.GetSpacing())
    return ct_masked


def resample(resample_factor, ori_img, resample_method=sitk.sitkNearestNeighbor):
    ori_img_array = sitk.GetArrayFromImage(ori_img)
    ori_spacing = np.array(ori_img.GetSpacing())
    ori_size = np.array(ori_img_array.shape)
    tar_size = ori_size // resample_factor
    tar_spacing = ori_spacing * resample_factor
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)
    resampler.SetSize(tar_size.tolist())
    resampler.SetOutputSpacing(tar_spacing)
    if resample_method == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)
    resampled_img = resampler.Execute(ori_img)
    return resampled_img


def padding(img, padded_size):
    if len(img.shape) == 3:
        N, H, W = img.shape
        N_tag = H_tag = W_tag = 0
        N_pad = padded_size - N
        if N_pad % 2 == 1:
            N_pad += 1
            N_tag = 1
        N_pad_size_left = N_pad_size_right = N_pad // 2
        H_pad = padded_size - H
        if H_pad % 2 == 1:
            H_pad += 1
            H_tag = 1
        H_pad_size_left = H_pad_size_right = H_pad // 2
        W_pad = padded_size - W
        if W_pad % 2 == 1:
            W_pad += 1
            W_tag = 1
        W_pad_size_left = W_pad_size_right = W_pad // 2
        img = np.pad(img, ((int(N_pad_size_left - N_tag), int(N_pad_size_right)),
                           (int(H_pad_size_left - H_tag), int(H_pad_size_right)),
                           (int(W_pad_size_left - W_tag), int(W_pad_size_right))), 'constant')
    if len(img.shape) == 2:
        H, W = img.shape
        H_tag = W_tag = 0
        H_pad = padded_size - H
        if H_pad % 2 == 1:
            H_pad += 1
            H_tag = 1
        H_pad_size_left = H_pad_size_right = H_pad // 2
        W_pad = padded_size - W
        if W_pad % 2 == 1:
            W_pad += 1
            W_tag = 1
        W_pad_size_left = W_pad_size_right = W_pad // 2
        img = np.pad(img, ((int(H_pad_size_left - H_tag), int(H_pad_size_right)),
                           (int(W_pad_size_left - W_tag), int(W_pad_size_right))), 'constant')
    return img


def batch_preprocess(path, resample_factor):
    files = os.listdir(path)
    for file in files:
        img_files = os.listdir(path + '/' + file)
        ct_list = []
        mask_list = []
        for img_file in img_files:
            if 'postprocessing' not in img_file and 'resampled' not in img_file:
                ct_list.append(img_file)
            if 'postprocessing' in img_file:
                mask_list.append(img_file)
        for ct_item in tqdm(ct_list):
            name = ct_item[:ct_item.find('.')]
            for mask_item in mask_list:
                if name in mask_item:
                    break
            ct_path = path + '/' + file + '/' + ct_item
            mask_path = path + '/' + file + '/' + mask_item
            ct = sitk.ReadImage(ct_path)
            mask = sitk.ReadImage(mask_path)
            ori_img = mask_calculate(ct, mask)
            sitk.WriteImage(ori_img, SAVE_PATH + '/' + name + '.nii.gz')


def single_preprocess_mask(mask_128, resample_factor):
    resampled_img = resample(resample_factor, mask_128)
    resampled_img_array = sitk.GetArrayFromImage(resampled_img)
    resampled_img_array = padding(resampled_img_array, padded_size=mask_128.GetSize()[1] // resample_factor)
    return resampled_img_array


def batch_preprocess_mask(path, resample_factor):
    '''save the mask image as same as ct volume'''
    SAVE_PATH = '../Data/mask/128'
    files = os.listdir(path)
    ct_list = []
    for file in files:
        ct_list.append(file)
    for ct_item in tqdm(ct_list):
        name = ct_item.split('.')[0]
        ct_path = path + '/' + ct_item
        ct = sitk.ReadImage(ct_path)
        resampled_img = resample(resample_factor, ct)
        resampled_img_spacing = resampled_img.GetSpacing()
        resampled_img_array = sitk.GetArrayFromImage(resampled_img)
        # resampled_img_array = pad(resampled_img_array, padded_size=ct.GetSize()[1] // resample_factor)
        if resampled_img_array.shape[2] > 128:
            if resampled_img_array.shape[2] % 2 == 1:
                center = (resampled_img_array.shape[2] + 1) // 2 - 1
            elif resampled_img_array.shape[2] % 2 == 0:
                center = resampled_img_array.shape[2] // 2
            resampled_img_array = resampled_img_array[:, :, center - 64:center + 64]
        resampled_img = sitk.GetImageFromArray(resampled_img_array)
        resampled_img.SetSpacing(resampled_img_spacing)
        sitk.WriteImage(resampled_img, SAVE_PATH + '/' + name + '.nii.gz')
        # sitk.WriteImage(ori_img, SAVE_PATH + '/' + name + '.nii.gz')


def single_preprocess_2d(resample_factor):
    '''down sample x-ray image'''
    ct_path = '../original_data/project/data4_512_project_frontal3.nii.gz'
    ori_img = sitk.ReadImage(ct_path)
    resampled_img = resample(resample_factor, ori_img, resample_method=sitk.sitkNearestNeighbor)
    resampled_img_spacing = resampled_img.GetSpacing()
    resampled_img_array = sitk.GetArrayFromImage(resampled_img)
    resampled_img_array = padding(resampled_img_array, padded_size=256)
    resampled_img = sitk.GetImageFromArray(resampled_img_array)
    resampled_img.SetSpacing(resampled_img_spacing)
    sitk.WriteImage(resampled_img, 'data4_128_project_frontal3.nii.gz')


if __name__ == '__main__':
    resample_factor = 4
    path = '../original_data/mask'
