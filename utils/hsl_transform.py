import torch


def rgb_to_hsl(rgb):
    # rgb: (B,3,H,W)
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]

    maxc = torch.max(rgb, dim=1)[0]
    minc = torch.min(rgb, dim=1)[0]
    l = (maxc + minc) / 2.0

    s = torch.zeros_like(l)
    h = torch.zeros_like(l)

    mask = maxc != minc

    s[mask] = torch.where(
        l[mask] < 0.5,
        (maxc[mask] - minc[mask]) / (maxc[mask] + minc[mask]),
        (maxc[mask] - minc[mask]) / (2.0 - maxc[mask] - minc[mask])
    )

    rc = (maxc - r) / (maxc - minc + 1e-6)
    gc = (maxc - g) / (maxc - minc + 1e-6)
    bc = (maxc - b) / (maxc - minc + 1e-6)

    h[mask & (r == maxc)] = (bc - gc)[mask & (r == maxc)]
    h[mask & (g == maxc)] = 2.0 + (rc - bc)[mask & (g == maxc)]
    h[mask & (b == maxc)] = 4.0 + (gc - rc)[mask & (b == maxc)]

    h = (h / 6.0) % 1.0

    return torch.stack([h, s, l], dim=1)
