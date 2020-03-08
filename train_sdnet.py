import argparse
import json
from easydict import EasyDict
from models.SDNet import SDNet
import torch
from torch.utils.data import DataLoader, random_split
from sdnet_train.SDNetTrain import train
from utils.normalize import normalize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config", default='', help="Experiment configuration file", required=True)
    config = parser.parse_args().config

    with open(config) as json_file:
        conf = EasyDict(json.load(json_file)).params

    sdnet = SDNet(conf.num_a, conf.num_z, conf.device, conf.shape, conf.binary)

    p_images = torch.load(conf.loader_loc)
    p_images = p_images.to(conf.device)

    for idx in range(p_images.shape[0]):
        normalize(p_images[idx])

    if conf.model_src != "":
        sdnet.load_state_dict(torch.load(conf.model_src))

    n_train = int(conf.split_portion * len(p_images))
    train_set, val_set = random_split(p_images, lengths=(n_train, len(p_images) - n_train))
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=conf.shuffle)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, num_workers=conf.num_workers)
    data = train_loader, val_loader

    train(data, sdnet, conf.epochs, conf.model_dest, val_set, conf.kl_weight, conf.show_every, conf.save_model, conf.lr)

