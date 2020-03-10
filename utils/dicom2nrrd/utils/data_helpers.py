import subprocess
from pathlib import Path
import time
import os
from utils import settings as S
import scipy.io as spio
import pickle
import numpy as np
import json
import SimpleITK as sitk
import matplotlib.pyplot as plt

epsilon = 1e-07


def save_pickle(p, object_to_save):
    with open(str(p), 'wb') as f:
        pickle.dump(object_to_save, f)


def read_pickle(p):
    with p.open('rb') as fp:
        data = pickle.load(fp)
    return data


def read_json(j):
    with j.open('r') as fp:
        data = json.load(fp)
    return data


def save_json(p, data):
    with p.open('w') as fp:
        json.dump(data, fp, indent=2)
    return 1


def normalize_0_1(a):
    return (a - np.min(a)) / np.ptp(a)


def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


def create_project_tree(folders_list):
    for folder in folders_list:
        Path(folder).mkdir(exist_ok=True)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def resample_new_spacing(image, target_spacing, interplator=sitk.sitkLinear):
    resample = sitk.ResampleImageFilter()
    input_size = image.GetSize()
    pixel_type = image.GetPixelID()
    input_spacing = image.GetSpacing()
    input_spacing = np.round(input_spacing, 2)
    output_spacing = target_spacing
    output_size = [
        int(input_spacing[0] / output_spacing[0] * input_size[0]),
        int(input_spacing[1] / output_spacing[1] * input_size[1]),
        int(input_spacing[2] / output_spacing[2] * input_size[2]),
    ]
    resample.SetInterpolator(interplator)
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputPixelType(pixel_type)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


def get_case_names(path, format):
    all_files = list(path.glob("*" + format))
    all_cases = [f.stem.split('_')[0] for f in all_files]
    return set(all_cases)


def calculate_fft_from_RF(rf_all_frames, fs, n_frames=100):
    rf_all_frames = rf_all_frames * np.hamming(rf_all_frames.shape[-1])
    L = next_power_of_2(n_frames)
    dc_means = np.mean(rf_all_frames, -1)
    rf_all_frames_dc_removed = rf_all_frames - np.expand_dims(dc_means, -1)
    rf_fft = np.fft.fft(rf_all_frames_dc_removed, L, axis=-1)
    rf_fft_abs = np.abs(rf_fft[..., 0:(L // 2 + 1)])
    rf_fft_abs /= n_frames
    zero_freq_terms = rf_fft_abs[..., 0]
    niquist_freq_terms = rf_fft_abs[..., -1]
    final_fft = 2 * rf_fft_abs
    final_fft[..., 0] = zero_freq_terms
    final_fft[..., -1] = niquist_freq_terms
    f = np.arange(0, (L // 2 + 1)) * fs / L
    return final_fft, f, dc_means


def get_dict_from_model_chronicle(model_id, dict_name):
    model_dir = S.train_folder / model_id
    model_chronicle_dir = model_dir / "chronicle"
    model_datagen_dict_path = model_chronicle_dir / (dict_name + ".json")
    return read_json(model_datagen_dict_path)


def get_datagen_dict_from_model_id(model_id):
    model_dir = S.train_folder / model_id
    model_chronicle_dir = model_dir / "chronicle"
    model_datagen_dict_path = model_chronicle_dir / "datagen_ae.json"
    return read_json(model_datagen_dict_path)


def resize_lossless(arr, final_size):
    s1, s2, s3 = arr.shape

    # diff in size
    d1 = final_size[0] - s1
    d2 = final_size[1] - s2

    if d1 > 0:
        # if smaller -> pad
        p1 = d1
        c1 = 0
    else:
        # if bigger -> cut
        p1 = 0
        c1 = -d1

    if d2 > 0:
        p2 = d2
        c2 = 0
    else:
        p2 = 0
        c2 = -d2

    padded_mask = np.pad(arr, [(p1, 0), (p2, 0), (0, 0)])
    final_mask = padded_mask[c1:, c2:, :]
    return final_mask


def create_data_dict_from_model_id(model_id, phase):
    model_datagen_dict = get_datagen_dict_from_model_id(model_id)
    model_split_id = model_datagen_dict["datagen"]["split_folder_uid"]
    data_dict = {}
    cores = read_pickle(S.split_folder / model_split_id / (phase + '.pickle'))
    data_dict["rf_filenames"] = cores
    data_dict["maximum_freq"] = model_datagen_dict["datagen"]["specs"]["maximum_freq"]
    data_dict["n_samples"] = model_datagen_dict["datagen"]["specs"]["n_samples"]
    return data_dict


def create_data_dict(datagen_dict, phase):
    data_dict = {}
    cores = read_pickle(S.split_folder / datagen_dict["split_folder_uid"] / (phase + '.pickle'))
    # sheets_dir = Path(S.sheets_folder)
    # df = pd.read_excel(sheets_dir / "PatientsInfo_All.xlsx")
    # rf_file_names = df["Filename"].tolist()
    # if datagen_dict["specs"]["remove_needle_cores"]:
    #     rf_file_names = [rf for rf in rf_file_names if not rf.startswith('rf201306')]
    data_dict["rf_filenames"] = cores
    return data_dict


def window_intensity(image, bounding_box=[0, 1, 0.1, 0.9, 0.1, 0.9], pl=1.0, ph=99.0):
    rescaler = sitk.IntensityWindowingImageFilter()
    nda = sitk.GetArrayFromImage(image)
    a = int(nda.shape[0] * bounding_box[0])
    b = int(nda.shape[0] * bounding_box[1])
    c = int(nda.shape[1] * bounding_box[2])
    d = int(nda.shape[1] * bounding_box[3])
    e = int(nda.shape[2] * bounding_box[4])
    f = int(nda.shape[2] * bounding_box[5])
    val_pl = np.percentile(nda[a:b, c:d, e:f], pl)
    val_ph = np.percentile(nda[a:b, c:d, e:f], ph)
    rescaler.SetWindowMaximum(val_ph)
    rescaler.SetWindowMinimum(val_pl)
    rescaler.SetOutputMaximum(val_ph)
    rescaler.SetOutputMinimum(val_pl)
    return rescaler.Execute(image)


def window_intensity2d(image, bounding_box=[0.1, 0.9, 0.1, 0.9], pl=1.0, ph=99.0):
    rescaler = sitk.IntensityWindowingImageFilter()
    nda = sitk.GetArrayFromImage(image)
    a = int(nda.shape[0] * bounding_box[0])
    b = int(nda.shape[0] * bounding_box[1])
    c = int(nda.shape[1] * bounding_box[2])
    d = int(nda.shape[1] * bounding_box[3])
    val_pl = np.percentile(nda[a:b, c:d], pl)
    val_ph = np.percentile(nda[a:b, c:d], ph)
    rescaler.SetWindowMaximum(val_ph)
    rescaler.SetWindowMinimum(val_pl)
    rescaler.SetOutputMaximum(val_ph)
    rescaler.SetOutputMinimum(val_pl)
    return rescaler.Execute(image)


def preprocess_one(image, prep_dict, mode="2d"):
    window_dict = prep_dict["window_intensity"]
    normalization = prep_dict["normalization"]
    #
    if window_dict["status"]:
        pl = window_dict["pl"]
        ph = window_dict["ph"]
        if mode == '2d':
            bounding_box = window_dict["bounding box"]
            image = window_intensity2d(image, bounding_box=bounding_box, pl=pl, ph=ph)

    if normalization["status"]:
        normalization_type = normalization["type"]
        if normalization_type == "zscore":
            f = sitk.NormalizeImageFilter()
            image = f.Execute(image)
        elif normalization_type == "scale":
            image = rescale_zero_one(image)

    return image


def rescale_zero_one(image):
    image = sitk.RescaleIntensity(sitk.Cast(image, sitk.sitkFloat32), 0, 1)
    return image


def plot_img2d(img):
    arr = getArr(img)
    plt.imshow(arr, cmap='gray')
    plt.colorbar()
    plt.show()


def lesion_probmap_itk(distance_map, sigma=3):
    distance = sitk.Square(distance_map)
    distance = sitk.Multiply(distance, 1 / (2 * np.square(sigma)))
    probmap = sitk.Multiply(sitk.ExpNegative(distance), 1 / np.sqrt(2 * np.pi * np.square(sigma)))
    return probmap


def lesion_probmap_numpy(distance_map, sigma=3):
    distance = distance_map ** 2
    distance /= (2 * np.square(sigma))
    probmap = np.exp(-distance) / (np.sqrt(2 * np.pi * np.square(sigma)))
    probmap = np.clip(probmap, epsilon, 1 - epsilon)
    return probmap


def get_prob_and_label_2d(image, mask, loc, dilation=15):
    outside = 0
    finding_nda = np.zeros(image.GetSize()).astype(np.uint8)
    finding_image = sitk.GetImageFromArray(finding_nda)
    finding_image.CopyInformation(image)
    # dilation
    finding_image_dilated = sitk.BinaryDilate(finding_image, (dilation, dilation, 0))
    finding_image_dilated = sitk.Mask(finding_image_dilated, mask, outsideValue=0)
    # gaussian
    distance_map = sitk.DanielssonDistanceMap(finding_image, useImageSpacing=True)
    # distance_ma = sitk.Mask(distance_map, mask, outsideValue=outside)
    # mask = sitk.BinaryThreshold(distance_map_masked, 0, 100)
    probmap = lesion_probmap_itk(distance_map)
    # finding_probmap = sitk.Mask(probmap, mask, outsideValue=outside)
    return finding_image_dilated, probmap


def find_dict_in_listOFdicts(ListOFdicts, key, value):
    indx = []
    for i in range(len(ListOFdicts)):
        d = ListOFdicts[i]
        if d[key] == value:
            indx.append(i)
    return indx


def myShow(arr):
    plt.imshow(arr * 1)
    plt.colorbar()
    plt.show()


def getArr(img):
    return sitk.GetArrayFromImage(img)


def flip_augment(arr):
    return np.concatenate((arr, np.flip(arr, axis=2)), axis=0)


def extract_slice_itk(image, slice_n):
    size = list(image.GetSize())
    size[2] = 0
    index = [0, 0, int(slice_n)]
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    image_slice = Extractor.Execute(image)
    return image_slice


def crop_uniform_spacing(image, x_center, y_center, output_size=256):
    image_size = image.GetSize()
    crop_filter = sitk.CropImageFilter()

    lower_b = x_center - output_size // 2, y_center - output_size // 2, 0
    upper_b = image_size[0] - x_center - output_size // 2, image_size[1] - y_center - output_size // 2, 0

    crop_filter.SetUpperBoundaryCropSize(upper_b)
    crop_filter.SetLowerBoundaryCropSize(lower_b)
    image = crop_filter.Execute(image)
    return image


def transform_image(image, transform, reference_image, default_value=None, interpolator=sitk.sitkLinear):
    if default_value == None:
        default_value = reference_image[0, 0, 0]
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def resample_target_size(image, target_size, interplator=sitk.sitkLinear):
    resample = sitk.ResampleImageFilter()
    input_size = image.GetSize()
    target_size = list(map(int, target_size))
    pixel_type = image.GetPixelID()
    input_spacing = image.GetSpacing()
    input_spacing = np.round(input_spacing, 2)
    output_spacing = [
        input_spacing[0] / target_size[0] * input_size[0],
        input_spacing[1] / target_size[1] * input_size[1],
        input_spacing[2] / target_size[2] * input_size[2],
    ]
    resample.SetInterpolator(interplator)
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(target_size)
    resample.SetOutputPixelType(pixel_type)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    return resample.Execute(image)


def get_center_lps(image):
    return image.TransformContinuousIndexToPhysicalPoint(
        tuple(np.array(image.GetSize()) / 2)
    )


def get_rigid_transformation_from_euler(rotation, translation, rotation_center=(0, 0, 0)):
    rotation_center = list(map(float, rotation_center))
    theta_x = rotation[0]
    theta_y = rotation[1]
    theta_z = rotation[2]
    tx = sitk.Euler3DTransform()
    tx.SetCenter(rotation_center)
    tx.SetRotation(theta_x, theta_y, theta_z)
    tx.SetTranslation(list(map(float, translation)))
    return tx


def get_dict_from_mat_file(mat_file_path):
    output_dict = {}
    mat_file = spio.loadmat(mat_file_path)
    mat_variables = [k for k in mat_file.keys() if not k.startswith('__')]
    for k in mat_variables:
        output_dict[k] = mat_file[k]
    return output_dict


def n4biasfieldcorrection(input_path, output_path):
    cmd = list()
    cmd.append(os.path.join(S.slicer_dir, "Slicer"))
    cmd.append("--launch")
    cmd.append(
        os.path.join(
            S.slicer_dir, "lib", "Slicer-4.10", "cli-modules", "N4ITKBiasFieldCorrection"
        )
    )
    cmd.append(input_path)
    cmd.append(output_path)
    subprocess.call(cmd)
