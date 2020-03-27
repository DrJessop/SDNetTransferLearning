import SimpleITK as sitk
import os
import numpy as np
from utils.normalize import normalize


if __name__ == "__main__":

    os.chdir("/home/andrewg/PycharmProjects/merged_data/t2_full/KGH")
    average_kgh = "/home/andrewg/PycharmProjects/merged_data/t2_full/average_kgh.nrrd"

    if average_kgh not in os.listdir():
        im = np.zeros((160, 160))
        num_additions = 0
        for image in os.listdir():
            sitk_im = sitk.ReadImage(image, sitk.sitkInt16)
            for dim in range(sitk_im.GetSize()[2]):
                im = im + sitk.GetArrayFromImage(sitk_im[:, :, dim])
                num_additions += 1

        im = im / num_additions
        im = sitk.GetImageFromArray(im)
        im.SetSpacing((0.5, 0.5))
        sitk.WriteImage(im, average_kgh)

    ref_im = sitk.ReadImage(average_kgh, sitk.sitkFloat64)

    os.chdir("/home/andrewg/PycharmProjects/merged_data/t2_full/ProstateX")
    target = "/home/andrewg/PycharmProjects/merged_data/t2_full/hist_equalized_t2"
    hist_match_filter = sitk.HistogramMatchingImageFilter()
    for im3t_file in os.listdir():
        im3t = sitk.ReadImage(im3t_file, sitk.sitkFloat64)
        slices = []
        for dim in range(im3t.GetSize()[2]):
            slice = hist_match_filter.Execute(im3t[:, :, dim], ref_im)
            slice = sitk.GetArrayFromImage(slice)
            normalize(slice)
            slices.append(slice)
        slices = np.array(slices)
        slices = sitk.GetImageFromArray(slices)
        slices.CopyInformation(im3t)

        sitk.WriteImage(slices, "/home/andrewg/PycharmProjects/merged_data/t2_full/hist_equalized_t2/{}".format(
           im3t_file
        ))


