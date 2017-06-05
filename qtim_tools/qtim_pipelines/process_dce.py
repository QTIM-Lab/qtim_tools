""" TODO: Delete this file once our DCE pipeline is set up.
"""

import DCE_Property_Calculator

"""
Put the paths to whatever files you are analyzing in this section. If you don't need a certain file, you can leave it as ''. 

If you only want to calculate properties in a certain ROI, put the filepath to that ROI in label_filepath.

If you want to take your AIF from a ROI of, say, and artery, then put the filepath to that ROI in AIF_filepath.

If you want to use a T1Map to calculate ktrans, ve, and AUC, put that filepath under T1_map_filepath.

"""

filepath = 'dce1_mc_ss.nii.gz'
label_filepath = ''
AIF_filepath = ''
T1_map_filepath = ''


"""
If your files don't have a specific name, you can instead provide suffixes. For example, say your filepath is dce1_ss.nii.gz. If you know your ROI is named dce1_ss-label.nii.gz, then you can grab it by changing the label_suffix parameter to "-label". The same goes for the other parameters.

"""

label_suffix = ''
AIF_suffix = ''
T1_map_suffix = ''

"""
Set your parameters for the DCE processing here. 

T1_tissue and T1_blood are for if you are using static T1 values instead of a T1Map. T1_blood should be higher than T1_tissue.

Relaxivity is specific to the contrast agent used. I think ours is usually .0039.

If you don't know the proper hematocrit, keep it at .45. I don't think it matters as much as the other parameters.

TR, TE, scan_time_seconds, injection_start_time_seconds, and flip_angle_degrees are essential. You can find them from a DICOM header.
"""

T1_tissue=1000
T1_blood=1440
relaxivity=.0045
hematocrit=0.45
TR=5
TE=2.1
scan_time_seconds=360
injection_start_time_seconds=60
flip_angle_degrees=30

"""
The program will skip over pixels that are equal to the mask_value in the first time point. This can be useful for skull-stripped images - otherwise, it would try and calculate over empty space.

Alternatively, you can speciy a mask_threshold. This means that all voxels with an intensity below mask_threshold at time point zero would be skipped. It is set to 0 here, so only negative voxels will be skipped.
"""

mask_value=0
mask_threshold=0

"""
Specify your output prefix here. The results will be three files titled "[prefix]ktrans.nii.gz", "[prefix]ve.nii.gz", and ''[prefix]auc.nii.gz".
AUC is currently not being calculated, so that will be an empty volume.
"""

outfile_prefix='dce_'

"""
For now, don't worry about these parameters.
"""

AIF_mode = 'label_average'
intial_fitting_function_parameters = [1,1]
default_population_AIF = False
label_mode = 'separate'
AIF_values_file = ''
label_value=1
AIF_label_value=1
param_file=[]
outputs=['ktrans','ve','auc']

if __name__ == '__main__':
	DCE_Property_Calculator.calc_DCE_properties_single(filepath, T1_tissue=T1_tissue, T1_blood=T1_blood, relaxivity=relaxivity, TR=TR, TE=TE, scan_time_seconds=scan_time_seconds, hematocrit=hematocrit, injection_start_time_seconds=injection_start_time_seconds, flip_angle_degrees=flip_angle_degrees, label_file=label_file, label_suffix=label_suffix, label_value=label_value, mask_value=mask_value, mask_threshold=mask_threshold, T1_map_file=T1_map_file, T1_map_suffix=T1_map_suffix, AIF_label_file=AIF_filepath,  AIF_values_file=AIF_values_file, AIF_mode=AIF_mode, AIF_label_suffix=AIF_suffix, AIF_label_value=AIF_label_value, label_mode=label_mode, param_file=param_file, default_population_AIF=default_population_AIF, intial_fitting_function_parameters=intial_fitting_function_parameters, outputs=outputs, outfile_prefix=outfile_prefix)