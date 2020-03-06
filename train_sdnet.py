import argparse
import json
from easydict import EasyDict
from models.SDNet import SDNet
import torch
from torch.utils.data import DataLoader, random_split
from utils.SDNetTrain import train_sdnet


class GetConfigurations:

    def __init__(self):
        self.args = GetConfigurations.read_params()

    @staticmethod
    def read_params():
        parser = argparse.ArgumentParser(description='')
        parser.add_argument("--config", default='', help="Experiment configuration file", required=True)
        return parser.parse_args()


if __name__ == "__main__":
    config = GetConfigurations()
    config = config.read_params().config
    with open(GetConfigurations().read_params().config) as json_file:
        conf = EasyDict(json.load(json_file)).params

    sdnet = SDNet(conf.num_a, conf.num_z, conf.device, conf.shape, conf.binary)

    p_images = torch.load(conf.loader_loc)
    p_images = p_images.to(conf.device)

    if conf.model_src != "":
        sdnet.load_state_dict(torch.load(conf.model_src))

    n_train = int(conf.split_portion * len(p_images))
    train, val = random_split(p_images, lengths=(n_train, len(p_images) - n_train))
    train = DataLoader(train, batch_size=conf.batch_size, num_workers=conf.num_workers)
    val = DataLoader(val, batch_size=conf.batch_size, num_workers=conf.num_workers)

    train_sdnet(p_images, sdnet, conf.epochs, conf.model_dest, conf.kl_weight, conf.show_every, conf.save_model,
                conf.lr)


