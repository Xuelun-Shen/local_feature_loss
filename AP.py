# -*- coding: utf-8 -*-
# @Time    : 2019/11/25 15:55
# @Author  : xylon

import torch


def AveragePrecision(x):
    """
    calculate average precision for x
    :param x:   (tensor) 2d dims: Bxd
                like:   [[1, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0, 0, 0, 0]]
    :return:    (tensor) scalar
                like:   0.92, 0.59, 0.81
    This example comes from the paper "Hashing as Tie-Aware Learning to Rank"(Figure 1).
    """
    assert len(x.shape) == 2
    device = x.device
    n = x.size(1)
    acum = torch.cumsum(x, 1)
    dcum = torch.arange(1, n + 1, dtype=torch.float, device=device)
    accu = acum.float() / dcum
    accu = accu * x.float()
    ap = accu.sum(dim=1) / x.sum(dim=1).float()
    return ap
