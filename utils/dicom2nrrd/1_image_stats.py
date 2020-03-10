import os
import pydicom
import pandas as pd
from tqdm import tqdm
from glob import glob
from utils import settings as S
from pathlib import Path

if __name__ == '__main__':
    dicom_root = Path(S.raw_folder)
    patient_dirs = list(dicom_root.glob("*PCAD*"))
    output_csv_path = os.path.join(S.sheets_folder, "mr_image_stats.csv")
    d = []
    for patient_dir in tqdm(sorted(patient_dirs)):
        study_dirs = list(patient_dir.glob("*"))
        # assert len(study_folders) == 1, "More than one study folder"
        for study_dir in study_dirs:
            series_dirs = list(study_dir.glob("*"))
            for series_dir in series_dirs:
                files = list(series_dir.glob("*"))
                files = [str(f) for f in files if "v_headers" not in str(f)]
                n_files = len(files)
                if n_files == 0:
                    print(series_dir)
                plan = pydicom.read_file(files[0])
                try:
                    patient_id = int(plan.PatientID)
                except:
                    patient_id = ''
                try:
                    manufacturer = plan.Manufacturer
                except:
                    manufacturer = ''
                try:
                    study_uid = plan.StudyInstanceUID
                except:
                    study_uid = ''
                try:
                    series_uid = plan.SeriesInstanceUID
                except:
                    series_uid = ''
                try:
                    study_description = plan.StudyDescription
                except:
                    study_description = ''
                try:
                    series_description = plan.SeriesDescription
                except:
                    series_description = ''
                try:
                    modality = plan.Modality
                except:
                    modality = ''
                try:
                    strength = plan.MagneticFieldStrength
                except:
                    strength = ''
                try:
                    contrast = plan.ContrastBolusAgent
                except:
                    contrast = ''
                try:
                    institution_name = plan.InstitutionName
                except:
                    institution_name = ''

                try:
                    station = plan.StationName
                except:
                    station = ''
                try:
                    model_name = plan.ManufacturerModelName
                except:
                    model_name = ''
                try:
                    series_number = plan.SeriesNumber
                except:
                    series_number = ''
                try:
                    referring_physician = plan.ReferringPhysicianName
                except:
                    referring_physician = ''
                try:
                    coil_name = plan.ReceiveCoilName
                except:
                    coil_name = ''
                try:
                    slice_thickness = float(plan.SliceThickness)
                except:
                    slice_thickness = ''
                try:
                    pixel_spacing = [float(e) for e in plan.PixelSpacing]
                except:
                    pixel_spacing = ''
                d.append({'mri_id': patient_dir,
                          'study_uid': study_uid,
                          'series_uid': series_uid,
                          'series_number': series_number,
                          'slice_tickness': slice_thickness,
                          'pixel_spacing': pixel_spacing,
                          'n_files': n_files,
                          'study_description': study_description,
                          'series_description': series_description,
                          'modality': modality,
                          'strength': strength,
                          'manufacturer': manufacturer,
                          'station': station,
                          'model_name': model_name,
                          'coil_name': coil_name,
                          })
    pd.DataFrame(d, columns=list(d[0].keys())).to_csv(output_csv_path, index=False)
