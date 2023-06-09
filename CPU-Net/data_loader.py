import os
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

class Dataset(Data.Dataset):
    def __init__(self, data_root, data_label):
        # 初始化
        self.data = data_root
        self.label = data_label

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        ctindex = self.label[index]
        # 返回值自动转换为torch的tensor类型
        return data, ctindex