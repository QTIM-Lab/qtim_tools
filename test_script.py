# import qtim_tools

# qtim_tools.qtim_dce.tofts_parametric_mapper.test_method_2d()

import os
import numpy as np

import qtim_tools

from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy, save_numpy_2_nifti
from qtim_tools.qtim_preprocessing.threshold import crop_with_mask
from qtim_tools.qtim_preprocessing.normalization import zero_mean_unit_variance


# label_numpy = nifti_2_numpy('C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T2_SS_mask.nii.gz')

for file in ['C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T2_Resample.nii.gz', 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_FLAIR_R_Resample.nii.gz', 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T1C_R_Resample.nii.gz']:
    file = 'C:/Users/abeers/Documents/GitHub/Public_QTIM/qtim_tools/qtim_tools/test_data/test_data_features/MRHead.nii.gz'
    input_numpy = nifti_2_numpy(file)
    input_numpy = file
    # masked_numpy = crop_with_mask(input_numpy, label_numpy)
    norm_numpy = zero_mean_unit_variance(input_numpy)
    print norm_numpy.shape
    print np.ptp(norm_numpy)

#     save_numpy_2_nifti(norm_numpy, file, 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/' + '_'.join(str.split(os.path.basename(file), '_')[0:2]) + '_preproc.nii.gz')


# from qtim_tools.qtim_dce.tofts_parametric_mapper import calc_DCE_properties_single

# filepath = 'C:/Users/abeers/Documents/Data/F2F_DEMOS/DCE_MRI_Test_Vol.nii.gz'
# ROI = 'C:/Users/abeers/Documents/Data/F2F_DEMOS/DCE_MRI_Test_Vol-label-small.nii.gz'
# AIF = 'C:/Users/abeers/Documents/Data/F2F_DEMOS/DCE_MRI_Test_Vol-AIF-label.nii.gz'

# calc_DCE_properties_single(filepath, T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=3.8, TE=2.1, scan_time_seconds=(3*64), hematocrit=0.45, injection_start_time_seconds=24, flip_angle_degrees=25, label_file=ROI, label_suffix='-label', label_value=1, mask_value=0, mask_threshold=0, T1_map_file=[], T1_map_suffix='-T1Map', AIF_label_file=AIF,  AIF_value_data=[], AIF_value_suffix=[], convert_AIF_values=True, AIF_mode='label_average', AIF_label_suffix=[], AIF_label_value=1, label_mode='separate', param_file=[], default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outputs=['ktrans','ve','auc'], outfile_prefix='', processes=1, gaussian_blur=.65, gaussian_blur_axis=2)