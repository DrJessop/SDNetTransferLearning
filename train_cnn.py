from utils.config import get_config
from utils.normalize import normalize
from utils.make_balanced_sampler import make_sampler
from models.CAM_CNN import CAM_CNN
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from classifier_train.convnet_train import train


class PID(Dataset):
    def __init__(self):
        self.length = n_range

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        start_pt = item*conf.num_crops
        return torch.Tensor([range(start_pt, start_pt + conf.num_wanted)])


if __name__ == "__main__":
    conf = get_config()

    seed = conf.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cnn = CAM_CNN(shape=conf.shape, num_channels=conf.num_channels).to(conf.device)

    p_images = torch.load(conf.images_loader).to(conf.device)
    p_labels = torch.load(conf.labels_loader).to(conf.device).long()

    hold_out = conf.hold_out_last

    p_images_hold_out = p_images[-(hold_out*conf.num_crops):]
    p_labels_hold_out = p_labels[-(hold_out*conf.num_crops):]
    p_images = p_images[:-(hold_out*conf.num_crops)]
    p_labels = p_labels[:-(hold_out*conf.num_crops)]

    for idx in range(p_images.shape[0]):
        normalize(p_images[idx])

    n_train = int((conf.split_portion*len(p_images))//conf.num_crops)
    n_range = p_images.shape[0]//conf.num_crops

    train_set, val_set = random_split(PID(), lengths=(n_train, n_range - n_train))

    if conf.shuffle:
        shuffle = True
        sampler = None
    else:
        shuffle = False
        sampler = make_sampler(train_set, p_labels)

    train_loader = DataLoader(train_set, batch_size=conf.batch_size, num_workers=conf.num_workers, sampler=sampler,
                              shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, num_workers=conf.num_workers)

    if conf.model_src != "":
        cnn.load_state_dict(torch.load(conf.model_src))

    train((train_loader, val_loader), p_images, p_labels, cnn, conf.model_dest, conf.epochs, conf.lr, conf.weight_decay,
          conf.save_model)
    torch.save(p_images_hold_out, conf.hold_out_images_dest)
    torch.save(p_labels_hold_out, conf.hold_out_labels_dest)
