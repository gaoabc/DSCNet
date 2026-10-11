# utils/hsic.py
import torch
import torch.nn as nn

def _rbf_kernel(x, sigma):

    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (B,1)
    dist = x_norm + x_norm.t() - 2 * torch.matmul(x, x.t())
    dist = torch.clamp(dist, min=0)
    K = torch.exp(-dist / (2 * sigma * sigma))
    return K


def hsic(x, y, sigma=1.0):

    B = x.size(0)
    H = torch.eye(B, device=x.device) - torch.ones((B, B), device=x.device) / B

    Kx = _rbf_kernel(x, sigma)
    Ky = _rbf_kernel(y, sigma)

    Kc = H @ Kx @ H
    Lc = H @ Ky @ H

    hsic_val = torch.trace(Kc @ Lc) / ((B - 1) ** 2 + 1e-8)
    return hsic_val


class HSIC_IB_Loss(nn.Module):

    def __init__(self, sigma=1.3, delta=0.7):
        super().__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, x, b, y):

        # Flatten features
        x_flat = x.mean(dim=[2, 3]) if x.dim() == 4 else x
        b_flat = b.mean(dim=[2, 3]) if b.dim() == 4 else b
        y_flat = y if y.dim() == 2 else y.view(y.size(0), -1)


        hsic_xb = hsic(x_flat, b_flat, sigma=self.sigma)
        hsic_by = hsic(b_flat, y_flat, sigma=self.sigma)

        loss_ib = hsic_xb - self.delta * hsic_by
        return loss_ib
