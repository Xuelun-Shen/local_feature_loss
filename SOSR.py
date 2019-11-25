# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 18:22
# @Author  : xylon
import torch

from exceptions import EmptyTensorError


def SOS_KNN_ORI_HOR(xa, xp, knn):
    """
    Second Order Similarity KNN Original Horizontal
    :param xa: x anchor, nxd
    :param xp: x positive, nxd
    :param knn: int, like: 8
    :return:
    """
    if xa.shape[0] <= knn:
        raise EmptyTensorError
    assert xa.shape[0] == xp.shape[0]
    n = xa.shape[0]
    device = xa.device
    # (n, n) - [0, 2] negative cosine similarity 越接近于 0 越好
    d_xixj = -torch.mm(xa, xa.t()) + 1
    d_xixj = d_xixj + torch.eye(n).to(device) * 10
    knn_xixj = d_xixj <= d_xixj.topk(k=8, dim=1, largest=False)[0][:, -1][:, None]

    d_xipxjp = -torch.mm(xp, xp.t()) + 1
    d_xipxjp = d_xipxjp + torch.eye(n).to(device) * 10
    knn_xipxjp = d_xipxjp <= d_xipxjp.topk(k=8, dim=1, largest=False)[0][:, -1][:, None]

    clabel = torch.max(knn_xixj, knn_xipxjp)
    d_sos = (d_xixj - d_xipxjp) ** 2
    d_sos = (d_sos * clabel.float()).sum(dim=1).sqrt().mean()

    # dv_xixj = d_xixj.masked_select(clabel)
    # dv_xipxjp = d_xipxjp.masked_select(clabel)
    # d_sos = ((dv_xixj - dv_xipxjp) ** 2).mean() / 4
    return d_sos


def SOS_KNN_ORI_VER(xa, xp, knn):
    """
    Second Order Similarity KNN Original Vertical
    :param xa: x anchor
    :param xp: x positive
    :param knn: int (like, 8)
    :return:
    """
    if xa.shape[0] <= knn:
        raise EmptyTensorError
    assert xa.shape[0] == xp.shape[0]
    n = xa.shape[0]
    device = xa.device
    # (n, n) - [0, 2] negative cosine similarity 越接近于 0 越好
    d_xixj = -torch.mm(xa, xa.t()) + 1
    d_xixj = d_xixj + torch.eye(n).to(device) * 10
    knn_xixj = d_xixj <= d_xixj.topk(k=8, dim=0, largest=False)[0][-1][None, :]

    d_xipxjp = -torch.mm(xp, xp.t()) + 1
    d_xipxjp = d_xipxjp + torch.eye(n).to(device) * 10
    knn_xipxjp = d_xipxjp <= d_xipxjp.topk(k=8, dim=0, largest=False)[0][-1][None, :]

    clabel = torch.max(knn_xixj, knn_xipxjp)
    d_sos = (d_xixj - d_xipxjp) ** 2
    d_sos = (d_sos * clabel.float()).sum(dim=1).sqrt().mean()
    
    # dv_xixj = d_xixj.masked_select(clabel)
    # dv_xipxjp = d_xipxjp.masked_select(clabel)
    # d_sos = ((dv_xixj - dv_xipxjp) ** 2).mean() / 4
    return d_sos


def SOS_KNN_MOD_HOR(xa, xp, knn):
    """
    Second Order Similarity KNN Modified Horizontal
    :param xa: x anchor, nxd
    :param xp: x positive, nxd
    :param knn: int, like: 8
    :return:
    """
    if xa.shape[0] <= knn:
        raise EmptyTensorError
    assert xa.shape[0] == xp.shape[0]
    n = xa.shape[0]
    device = xa.device
    # (n, n) - [0, 2] negative cosine similarity 越接近于 0 越好
    d_xixj = -torch.mm(xa, xa.t()) + 1
    d_xixj = d_xixj + torch.eye(n).to(device) * 10
    knn_xixj = d_xixj <= d_xixj.topk(k=8, dim=1, largest=False)[0][:, -1][:, None]

    d_xipxjp = -torch.mm(xp, xp.t()) + 1
    d_xipxjp = d_xipxjp + torch.eye(n).to(device) * 10
    knn_xipxjp = d_xipxjp <= d_xipxjp.topk(k=8, dim=1, largest=False)[0][:, -1][:, None]

    clabel = torch.max(knn_xixj, knn_xipxjp)
    # d_sos = (d_xixj - d_xipxjp) ** 2
    # d_sos = (d_sos * clabel.float()).sum(dim=1).sqrt().mean()

    dv_xixj = d_xixj.masked_select(clabel)
    dv_xipxjp = d_xipxjp.masked_select(clabel)
    d_sos = ((dv_xixj - dv_xipxjp) ** 2).mean() / 4
    return d_sos


def SOS_KNN_MOD_VER(xa, xp, knn):
    """
    Second Order Similarity KNN Modified Vertical
    :param xa: x anchor
    :param xp: x positive
    :param knn: int (like, 8)
    :return:
    """
    if xa.shape[0] <= knn:
        raise EmptyTensorError
    assert xa.shape[0] == xp.shape[0]
    n = xa.shape[0]
    device = xa.device
    # (n, n) - [0, 2] negative cosine similarity 越接近于 0 越好
    d_xixj = -torch.mm(xa, xa.t()) + 1
    d_xixj = d_xixj + torch.eye(n).to(device) * 10
    knn_xixj = d_xixj <= d_xixj.topk(k=8, dim=0, largest=False)[0][-1][None, :]

    d_xipxjp = -torch.mm(xp, xp.t()) + 1
    d_xipxjp = d_xipxjp + torch.eye(n).to(device) * 10
    knn_xipxjp = d_xipxjp <= d_xipxjp.topk(k=8, dim=0, largest=False)[0][-1][None, :]

    clabel = torch.max(knn_xixj, knn_xipxjp)
    # d_sos = (d_xixj - d_xipxjp) ** 2
    # d_sos = (d_sos * clabel.float()).sum(dim=1).sqrt().mean()

    dv_xixj = d_xixj.masked_select(clabel)
    dv_xipxjp = d_xipxjp.masked_select(clabel)
    d_sos = ((dv_xixj - dv_xipxjp) ** 2).mean() / 4
    return d_sos
