import os
import SimpleITK as sitk
import torch
import numpy as np


def fetch_data(dir, HEIGHT, WIDTH):
    os.chdir(dir)
    data = torch.empty(0, 1, HEIGHT, WIDTH)
    for idx, f in enumerate(os.listdir()):
        slices = sitk.ReadImage(f)
        h, w, _ = slices.GetSize()
        if (h, w) != (HEIGHT, WIDTH):
            dh, dw = HEIGHT, WIDTH
            slices = slices[(h-dh)//2:(h+dh)//2]
            slices = slices[:, (w-dw)//2:(w+dw)//2]
        slices = sitk.GetArrayFromImage(slices).astype(np.float32)
        slices = torch.from_numpy(slices)
        slices = slices.reshape(slices.shape[0], 1, *slices.shape[1:])
        data = torch.cat((data, slices), dim=0)
    return data
