import sys
sys.path.append("..")
from utils.resample_image import resample_image
from utils.config import get_config
import os
import SimpleITK as sitk


if __name__ == "__main__":
    conf = get_config()
    path = conf.data_src
    dest = conf.data_dest

    files = os.listdir(path)
    dest_files = os.listdir(dest)

    for f in files:
        if f not in dest_files:
            g = sitk.ReadImage("{}/{}".format(path, f))
            g = resample_image(g, conf.spacing)
            sitk.WriteImage(g, "{}/{}".format(dest, f))
