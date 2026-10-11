import os
import cv2
import torch
from torch.utils.data import Dataset
from utils.hsl_transform import rgb_to_hsl


class GastroDataset(Dataset):
    def __init__(self, root, labeled_ratio=0.1, mode="train"):
        self.root = root
        self.mode = mode

        img_dir = os.path.join(root, "images")
        mask_dir = os.path.join(root, "masks")

        self.img_list = sorted(os.listdir(img_dir))
        self.mask_list = sorted(os.listdir(mask_dir))

        N = len(self.img_list)
        L = int(N * labeled_ratio)

        if mode == "train":
            self.img_list = self.img_list[:N]
        elif mode == "val":
            self.img_list = self.img_list[L:]

        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        img = cv2.imread(img_path)[:, :, ::-1] / 255.0
        mask = cv2.imread(mask_path, 0)

        img = torch.tensor(img).permute(2, 0, 1).float()
        mask = torch.tensor(mask).long()

        hsl = rgb_to_hsl(img)

        return {
            "image": img,
            "hsl": hsl,
            "mask": mask,
        }
