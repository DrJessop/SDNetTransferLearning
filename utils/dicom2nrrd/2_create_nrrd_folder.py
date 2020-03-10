import os
import shutil
from utils import settings as s
from utils.helpers import get_timestamp
from glob import glob
from utils.data_helpers import safe_mkdir

# first run dicom2nrrd notebook

dicom2nrrd_dir = os.path.join(s.raw_folder, "dicom2nrrd")
cases = os.listdir(dicom2nrrd_dir)

nrrd_output_dir = os.path.join(s.nrrd_folder, get_timestamp())
safe_mkdir(nrrd_output_dir)

excluded_cases = ["MRI628"]

for case in cases:
    if case not in excluded_cases:
        case_dir = os.path.join(dicom2nrrd_dir, case)
        study_id = os.listdir(case_dir)[0]
        case_images_dir = os.path.join(case_dir, study_id)
        files = glob(case_images_dir + '/*')
        images = [f for f in files if '.xml' not in f]
        assert len(images) == 2, "erronous case {}".format(case)
        for image in images:
            image_name = os.path.basename(image)
            if 'T2' in image_name:
                output_name = '_t2'
            else:
                output_name = '_adc'
            shutil.move(image, os.path.join(nrrd_output_dir,
                                            'case' + case.split('MRI')[-1].zfill(3) + output_name + '.nii.gz'))
