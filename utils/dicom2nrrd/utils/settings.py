from pathlib import Path
import socket
import pandas as pd
import os

folders = []
if socket.gethostname() == 'pmous008':
    intermediate_folder = Path("/home/sedghi/projects/kgh_mri")
    folders.append(intermediate_folder)
    raw_folder = Path("/DATA/hmrc_menard/AndrewKGH_NewData")
    slicer_dir = os.environ['HOME'] + "/sources/Slicer-4.10.2-linux-amd64"

# data folders
data_folder = intermediate_folder / 'data'
folders.append(data_folder)
sheets_folder = data_folder / 'sheets'
folders.append(sheets_folder)
split_folder = data_folder / 'split'
folders.append(split_folder)
npy_folder = data_folder / 'npy'
folders.append(npy_folder)
nrrd_folder = data_folder / 'nrrd'
folders.append(nrrd_folder)
eda_folder = data_folder / 'eda'
folders.append(eda_folder)
jpg_folder = eda_folder / 'jpg'
folders.append(jpg_folder)
roi_mask_folder = eda_folder / 'rf_rois_npy'
folders.append(eda_folder)
prostate_mask_folder = eda_folder / 'rf_prostate_masks'
folders.append(eda_folder)
# train folders
train_folder = intermediate_folder / 'train'
folders.append(train_folder)
predict_folder = intermediate_folder / 'predict'
folders.append(predict_folder)
# postprocess folder
postprocess_folder = intermediate_folder / 'postprocess'
folders.append(postprocess_folder)
