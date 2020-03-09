import argparse
import json
from easydict import EasyDict


def get_config():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--config", default='', help="Experiment configuration file", required=True)
    config = parser.parse_args().config

    with open(config) as json_file:
        conf = EasyDict(json.load(json_file)).params

    return conf

