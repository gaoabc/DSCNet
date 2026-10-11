
import torch
import random
import numpy as np
from torchvision.transforms import functional as TF
from utils.hsl_transform import rgb_to_hsl

class RGBToHSLTransform:
    def __call__(self, img):
        img = img.float() / 255.0
        return rgb_to_hsl(img)

class DualViewTransform:

    def __init__(self):
        pass

    def __call__(self, img):

        img = img.float()


        weak_img = TF.to_tensor(img)

        rgb = TF.to_tensor(img)
        hsl = rgb_to_hsl(rgb.unsqueeze(0)).squeeze(0)

        return weak_img, hsl
