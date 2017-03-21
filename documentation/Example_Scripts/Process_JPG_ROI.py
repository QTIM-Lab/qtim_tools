from qtim_tools.qtim_preprocessing.image import fill_in_convex_outline
from qtim_tools.qtim_utilities.nifti_util import replace_slice
from qtim_tools.qtim_preprocessing.registration import register_all_to_one
from qtim_tools.qtim_features import extract_features

import numpy as np

def Convert_JPG_ROI_2_Nifti(input_image, output_nifti, reference_nifti, ROI_color_range, slice_number):

    # This function takes the image defined by input_image, fills in closed lines in the color range defined by ROI_color_range,
    # and saves it to a new one-slice nifti file with a header defined by reference_nifti.
    fill_in_convex_outline(input_image, output_nifti, reference_nifti = reference_nifti, color_threshold_limits = ROI_color_range)

    # This function expands your one-slice nifti into a full 3D volume according to a provided reference Nifti volume. Slice number
    # is required to orient the original 2D volume in space.
    replace_slice(input_nifti_slice_filepath = output_nifti, reference_nifti_filepath = reference_nifti, output_file = output_nifti, slice_number = slice_number, orientation_commands=[np.rot90, np.flipud])

    return

def Register_Folder_to_One_Volume(fixed_volume, moving_volume_folder, output_folder, output_suffix, Slicer_Path):

    # Registers all volumes in a folder to one chosen fixed volume using Slicer's BRAINSFit registration.
    register_all_to_one(fixed_volume = fixed_volume, moving_volume_folder = moving_volume_folder, output_folder=output_folder, output_suffix=output_suffix, Slicer_Path=Slicer_Path)

    return

def Extract_Features_From_Folder(input_folder, output_csv, label_filepath):

    # Extract texture and shape features using QTIM's feature extraction package. Erode labels by 3 voxels in the XY plane.
    extract_features(input_folder, output_csv, set_label=label_filepath, erode=[3,3,0])

    return

def Full_Pipeline():

    # Parameters for JPG -> NIFTI conversion.
    input_image = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/image_with_roi.jpg'
    reference_nifti = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/7_Ax_T2_PROPELLER.nii.gz'
    output_nifti = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/image_with_roi-label.nii.gz'
    ROI_color_range = [[100,300],[0,100],[0,100]] # RGB -> Bright Red
    slice_number = 10
    # Convert_JPG_ROI_2_Nifti(input_image, output_nifti, reference_nifti, ROI_color_range, slice_number)

    # Parameters for group registration.
    Slicer_Path = '"C:/Users/azb22/Documents/Software/SlicerNightly/Slicer 4.6.0/Slicer.exe"'
    fixed_volume = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/7_Ax_T2_PROPELLER.nii.gz'
    moving_volume_folder = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/'
    output_folder = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/Registered_Volumes/'
    output_suffix = '_registered_T2'
    Register_Folder_to_One_Volume(fixed_volume, moving_volume_folder, output_folder, output_suffix, Slicer_Path)

    # Parameters for feature extraction.
    input_folder = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/Registered_Volumes/'
    output_csv = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/Feature_Extraction_Results.csv'
    label_filepath = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/image_with_roi-label.nii.gz'
    # Extract_Features_From_Folder(input_folder, output_csv, label_filepath)

if __name__ == "__main__":
    Full_Pipeline()