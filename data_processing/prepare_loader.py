import sys
sys.path.append("..")
from utils.config import get_config
from loaders.loader import fetch_data
import torch
import os


if __name__ == "__main__":
    conf = get_config()
    cur_dir = os.getcwd()
    data = fetch_data(conf.dir, *conf.shape)
    os.chdir(cur_dir)
    torch.save(data, conf.dest)
