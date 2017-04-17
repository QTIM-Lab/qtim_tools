from qtim_tools.qtim_preprocessing.image import fill_in_convex_outline
from qtim_tools.qtim_utilities.nifti_util import replace_slice, generate_identity_affine, save_numpy_2_nifti_no_reference
from qtim_tools.qtim_utilities.dicom_util import get_dicom_dictionary, dcm_2_numpy
from qtim_tools.qtim_utilities.array_util import generate_identity_affine
from qtim_tools.qtim_features import extract_features

from optparse import OptionParser

import numpy as np
import nibabel as nib
import glob
import os
import re

def Convert_JPG_ROI_2_Nifti(roi_folder, original_folder, output_nifti, output_nifti_label, ROI_color_range=[[200,300],[0,100],[0,100]]):

    original_dicom_dict = get_dicom_dictionary(original_folder)

    spacing = [original_dicom_dict['PixelSpacing'][0], original_dicom_dict['PixelSpacing'][1], original_dicom_dict['SpacingBetweenSlices']]
    spacing = [float(i) for i in spacing]

    original_numpy_data = dcm_2_numpy(original_folder)

    roi_numpy_data = np.zeros_like(original_numpy_data)

    ROI_list = sort_human(glob.glob(os.path.join(roi_folder, '*')))

    for ROI_idx, ROI in enumerate(ROI_list):
        roi_numpy_data[..., ROI_idx] = fill_in_convex_outline(ROI, color_threshold_limits = ROI_color_range)

    nifti_volume = save_numpy_2_nifti_no_reference(original_numpy_data)
    nifti_label = save_numpy_2_nifti_no_reference(roi_numpy_data)

    nifti_volume.header['pixdim'][1:4] = spacing  
    nifti_label.header['pixdim'][1:4] = spacing

    nib.save(nifti_volume, output_nifti)
    nib.save(nifti_label, output_nifti_label)

    return

def Extract_Features_From_Folder(input_folder, input_label, output_csv):

    # Extract texture and shape features using QTIM's feature extraction package. Erode labels by 3 voxels in the XY plane.
    extract_features(input_folder, output_csv, set_label=input_label, erode=[3,3,0])

    return

def Full_Pipeline(roi_folder, original_folder, output_nifti, output_nifti_label, feature_input_folder, output_features):

    Convert_JPG_ROI_2_Nifti(roi_folder, original_folder, output_nifti, output_nifti_label)
    Extract_Features_From_Folder(feature_input_folder, output_nifti_label, output_features)

def sort_human(l):
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
    l.sort( key=alphanum )
    return l

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--in_roi", "--roi_folder", dest="roi_folder", help="Folder containing JPG Segmentations to be translated into Nifti files")
    parser.add_option("--in_original", "--original_folder", dest="original_folder", help="Folder containing original DICOM files.")
    parser.add_option("--out_nifti", "--output_nifti", dest="output_nifti", help="Output Nifti file converted from the original DICOM files")
    parser.add_option("--out_label", "--output_nifti_label", dest="output_nifti_label", help="Output segmentation file converted from the JPG files")
    parser.add_option("--in_features", "--feature_folder", dest="feature_input_folder", help="Input folder for feature extraction")
    parser.add_option("--out_features", "--output_features_csv", dest="output_features", help="Output csv file for feature extraction")
    (options, args) = parser.parse_args()
    Full_Pipeline(options.roi_folder, options.original_folder, options.output_nifti, options.output_nifti_label, options.feature_input_folder, options.output_features)

# python Process_Multiple_JPG_ROI.py --roi_folder ".\subject data\Subject ROI marked" --original_folder ".\subject data\Subject original sequence\SE5" --output_nifti ".\subject data\test_subject.nii.gz" --output_nifti_label ".\subject data\test_subject-label.nii.gz" --feature_folder ".\subject data" --output_features_csv ".\subject data\output_features.csv" 