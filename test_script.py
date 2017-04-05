# import qtim_tools

# qtim_tools.qtim_dce.tofts_parametric_mapper.test_method_2d()

import os

from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy, save_numpy_2_nifti
from qtim_tools.qtim_preprocessing.threshold import crop_with_mask
from qtim_tools.qtim_preprocessing.normalization import zero_mean_unit_variance

label_numpy = nifti_2_numpy('C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T2_SS_mask.nii.gz')

for file in ['C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T2_Resample.nii.gz', 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_FLAIR_R_Resample.nii.gz', 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/TATA_T1C_R_Resample.nii.gz']:
    input_numpy = nifti_2_numpy(file)
    masked_numpy = crop_with_mask(input_numpy, label_numpy)
    norm_numpy = zero_mean_unit_variance(masked_numpy, mask_numpy=label_numpy)

    save_numpy_2_nifti(norm_numpy, file, 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/DeepMedic_Test/' + '_'.join(str.split(os.path.basename(file), '_')[0:2]) + '_preproc.nii.gz')
