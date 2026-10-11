import torch
import torch.nn as nn
import torch.nn.functional as F


class HSIC_IB_Loss(nn.Module):
    def __init__(self, sigma=1.0, lam=0.5):
        super().__init__()
        self.sigma = sigma
        self.lam = lam

    def rbf(self, x):
        x = x.flatten(1)
        dist = torch.cdist(x, x)
        return torch.exp(-dist**2 / (2 * self.sigma**2))

    def HSIC(self, X, Y):
        K = self.rbf(X)
        L = self.rbf(Y)

        H = torch.eye(K.shape[0]).cuda() - 1/K.shape[0]
        HKH = H @ K @ H
        HLH = H @ L @ H
        return torch.trace(HKH @ HLH)

    def forward(self, feat_rgb, feat_hsl, mask):
        return self.HSIC(feat_rgb, mask) - self.lam * self.HSIC(feat_hsl, mask)
