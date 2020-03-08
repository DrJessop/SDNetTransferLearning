"""
This experiment involves converting all 3.0 T images to 1.5 T images
"""
import torch
import json
import argparse
import os
from easydict import EasyDict
import SimpleITK as sitk
import numpy as np
import sys
sys.path.append("..")
from models.SDNet import SDNet
import shutil
from utils.normalize import normalize


def get_tensor_from_1p5t_data():
    """
    Converts 1.5 T nrrd file directories to torch tensors
    """
    data_1p5t = torch.empty(0, 1, *conf.shape).to(conf.device)
    for im in os.listdir(conf.data_1p5t):
        path = os.path.join(conf.data_1p5t, im)
        shutil.copyfile(path, os.path.join(conf.data_dest, im))
        image = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(image)
        image = torch.from_numpy(image.astype(np.float32)).to(conf.device)
        image = image.unsqueeze(1)
        data_1p5t = torch.cat((data_1p5t, image), dim=0)

    return data_1p5t


def convert():
    data_1p5t = get_tensor_from_1p5t_data()
    for idx in range(data_1p5t.shape[0]):
        normalize(data_1p5t[idx])
    num_ims = data_1p5t.shape[0]
    sdnet = SDNet(conf.num_a, conf.num_z, conf.device, conf.shape, conf.binary)
    sdnet.load_state_dict(torch.load(conf.model_src))

    for im in os.listdir(conf.data_3t):
        path = os.path.join(conf.data_3t, im)
        image3t_sitk = sitk.ReadImage(path)
        image3t = sitk.GetArrayFromImage(image3t_sitk)
        image3t = torch.from_numpy(image3t.astype(np.float32)).to(conf.device)
        for idx in range(image3t.shape[0]):
            normalize(image3t[idx])
        image3t = image3t.unsqueeze(1)
        image1p5t = data_1p5t[np.random.randint(0, num_ims)]
        shape_im = image1p5t.shape[0]
        index = torch.Tensor(np.random.choice(shape_im, image3t.shape[0])).long()
        image1p5t = image1p5t[index].unsqueeze(1)

        # Convert 3t to 1.5t
        with torch.no_grad():
            anatomy, _, _, _ = sdnet(image3t)
            _, _, _, z = sdnet(image1p5t)
            reconstruction = sdnet.decoder(anatomy, z).squeeze(1)

        reconstruction = reconstruction.cpu().numpy()
        reconstruction = sitk.GetImageFromArray(reconstruction)
        reconstruction.CopyInformation(image3t_sitk)
        path = os.path.join(conf.data_dest, im)
        sitk.WriteImage(reconstruction, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config", default='', help="Experiment configuration file", required=True)
    config = parser.parse_args().config

    with open(config) as json_file:
        conf = EasyDict(json.load(json_file)).params

    convert()

