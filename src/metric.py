import torch
from sklearn.metrics.cluster import mutual_info_score as MI
from torch import nn

from util import cal_ncc


# Gradient-NCC loss
def gradncc(I, J, device='cuda', win=None, eps=1e-10):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)

    return 1 - 0.5 * cal_ncc(Ix, Jx, eps) - 0.5 * cal_ncc(Iy, Jy, eps)


# MI loss
def mi(I, J):
    I = I.cpu().detach().numpy().flatten()
    J = J.cpu().detach().numpy().flatten()
    return 1 - MI(I, J)


# NGI loss
def ngi(I, J, device='cuda'):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)

    Ix = Ix.cpu().detach().numpy().flatten()
    Iy = Iy.cpu().detach().numpy().flatten()
    Jx = Jx.cpu().detach().numpy().flatten()
    Jy = Jy.cpu().detach().numpy().flatten()
    return 1 - 0.5 * MI(Ix, Jx) - 0.5 * MI(Iy, Jy)


# NCC loss
def ncc(I, J, device='cuda', win=None, eps=1e-10):
    return 1 - cal_ncc(I, J, eps)


# Cosine similarity
def cos_sim(a, b, device='cuda', win=None, eps=1e-10):
    return torch.sum(torch.multiply(a, b)) / ((torch.sum((a) ** 2) ** 0.5) * (torch.sum((b) ** 2)) ** 0.5 + eps)


# NCCL loss
def nccl(I, J, device='cuda', kernel_size=5, win=None, eps=1e-10):
    '''
    Normalized cross-correlation (NCCL) based on the LOG
    operator is obtained. The Laplacian image is obtained by convolution of the reference image
    and DRR image with the LOG operator. The zero-crossing point in the Laplacian image
    is no longer needed to obtain the image¡¯s detailed edge. However, two Laplacian images¡¯
    consistency is directly measured to use image edge and detail information effectively. This
    paper uses cosine similarity to measure the similarity between Laplacian images.
    '''

    # Compute filters
    with torch.no_grad():
        if kernel_size == 5:
            kernel_LoG = torch.Tensor([[[[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4], [-4, 0, 8, 0, -4],
                                         [-2, -4, -4, -4, -2]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 5, 1, 1, bias=False)
        elif kernel_size == 9:
            kernel_LoG = torch.Tensor([[[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [1, 4, 5, 3, 0, 3, 5, 4, 1],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [2, 5, 0, -24, -40, -24, 0, 5, 2],
                                         [2, 5, 3, -12, -24, -12, 3, 5, 2],
                                         [1, 4, 5, 3, 0, 3, 4, 4, 1],
                                         [1, 2, 4, 5, 5, 5, 4, 2, 1],
                                         [0, 1, 1, 2, 2, 2, 1, 1, 0]]]])
            kernel_LoG = torch.nn.Parameter(kernel_LoG, requires_grad=False)
            LoG = nn.Conv2d(1, 1, 9, 1, 1, bias=False)
        LoG.weight = kernel_LoG
        LoG = LoG.to(device)
    LoG_I = LoG(I)
    LoG_J = LoG(J)
    # Cosine_similarity
    return 1.5 - cal_ncc(I, J) - 0.5 * cos_sim(LoG_I, LoG_J)


# GD loss
def gradient_difference(I, J, s=1, device='cuda', win=None, eps=1e-10):
    # Compute filters
    with torch.no_grad():
        kernel_X = torch.Tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        kernel_X = torch.nn.Parameter(kernel_X, requires_grad=False)
        kernel_Y = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        kernel_Y = torch.nn.Parameter(kernel_Y, requires_grad=False)
        SobelX = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelX.weight = kernel_X
        SobelY = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        SobelY.weight = kernel_Y

        SobelX = SobelX.to(device)
        SobelY = SobelY.to(device)

    Ix = SobelX(I)
    Iy = SobelY(I)
    Jx = SobelX(J)
    Jy = SobelY(J)
    # Compute difference image
    if s != 1:
        Idx = Ix - s * Jx
        Idy = Iy - s * Jy
    else:
        Idx = Ix - Jx
        Idy = Iy - Jy
    # Compute variance of image
    N = torch.numel(Ix)
    Av = torch.sum((Ix - torch.mean(Ix)) ** 2) / N
    Ah = torch.sum((Iy - torch.mean(Iy)) ** 2) / N
    g = torch.sum(Av / (Av + (Idx) ** 2)) + torch.sum(Ah / (Ah + (Idy) ** 2))
    return 1 - 0.5 * g / N


# PWE loss
def pixel_wise_error(I, J):
    return torch.norm(I - J) / torch.prod(torch.tensor(I.size()))
