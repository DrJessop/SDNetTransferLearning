import os
import sys
sys.path.append("..")
from utils.config import get_config
from utils.rotation3d import rotation3d
from utils.image_cropper import crop_from_center
import pandas as pd
import SimpleITK as sitk


def write_crops():
    df = pd.read_csv(conf.dataframe_src)
    name = ""
    counter = 0
    for idx, row in df.iterrows():
        if row["anonymized"] != name:
            counter = 0
        name = row["anonymized"]
        clin_sig = row["ClinSig"]
        if clin_sig == "TRUE":
            clin_sig = 1
        elif clin_sig == "FALSE":
            clin_sig = 0
        else:
            clin_sig = int(int(clin_sig) > 0)
        path = os.path.join(conf.data_src, name) + ".{}".format(conf.extension)
        im = sitk.ReadImage(path)
        lps = row["lps"].split(',')
        lps = [float(s) for s in lps]
        ijk = im.TransformPhysicalPointToIndex(lps)

        max_angle = 2*conf.num_crops
        angles = [angle for angle in range(0, max_angle, 2)]
        file_name = "{}_{}".format(name, clin_sig)
        crops = []
        paths = []
        valid = True
        for angle in angles:
            rotated_image = rotation3d(im, angle, lps)
            crop_name = "{}_{}.{}".format(file_name, counter, conf.extension)
            counter += 1
            im_path = os.path.join(conf.data_dest, crop_name)
            crops.append(crop_from_center(rotated_image, ijk, *conf.sitk_size))
            if crops[-1].GetSize() != tuple(conf.sitk_size):
                valid = False
                break
            paths.append(im_path)

        if valid:
            for crop, path in zip(crops, paths):
                sitk.WriteImage(crop, path)


if __name__ == "__main__":
    conf = get_config()
    write_crops()

