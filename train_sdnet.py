from models.SDNet import SDNet
import torch
from torch.utils.data import DataLoader, random_split
from sdnet_train.SDNetTrain import train, test_slice
from utils.normalize import normalize
from utils.config import get_config
from random import randint as randint


if __name__ == "__main__":
    conf = get_config()

    seed = conf.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sdnet = SDNet(conf.num_a, conf.num_z, conf.device, conf.shape, conf.binary)

    p_images = torch.load(conf.loader_loc)
    p_images = p_images.to(conf.device)

    if conf.normalize:
        for idx in range(p_images.shape[0]):
            normalize(p_images[idx])

    if conf.model_src != "":
        sdnet.load_state_dict(torch.load(conf.model_src))

    n_train = int(conf.split_portion * len(p_images))
    train_set, val_set = random_split(p_images, lengths=(n_train, len(p_images) - n_train))
    train_loader = DataLoader(train_set, batch_size=conf.batch_size, num_workers=conf.num_workers, shuffle=conf.shuffle)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, num_workers=conf.num_workers)
    data = train_loader, val_loader

    if conf.train:
        train(data, sdnet, conf.epochs, conf.model_dest, val_set, conf.kl_weight, conf.show_every, conf.save_model,
              conf.lr)
    else:
        test_slice(randint(0, len(val_set)), val_set, sdnet)
