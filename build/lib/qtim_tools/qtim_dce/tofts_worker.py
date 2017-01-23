import tofts_parameter_calculator
# from ..qtim_utilities import nifti_util

def test(filepath=[]):
	tofts_parameter_calculator.test_method_3d(filepath)

def run_test():

	# You must install the packages nibabel and pydicom before running this program.

	filepath='C:/Users/abeers/Documents/GitHub/Public_QTIM/qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v6.nii.gz'
	
	label_file=[]
	label_suffix=[]
	label_value=1
	label_mode='separate'

	T1_map_file=[]
	T1_map_suffix=[]

	AIF_value_data=[]
	AIF_value_suffix=[]
	convert_AIF_values=False
	default_population_AIF=False
	
	AIF_label_file=[]
	AIF_mode='label_average'
	AIF_label_suffix='-AIF-label'
	AIF_label_value=1

	T1_tissue=1000
	T1_blood=1440
	relaxivity=.0045
	TR=5
	TE=2.1
	scan_time_seconds=(11*60)
	hematocrit=0.45
	injection_start_time_seconds=60
	flip_angle_degrees=30
	initial_fitting_function_parameters=[.3,.1]
	
	outputs=['ktrans','ve','auc']
	outfile_prefix='tofts_v6_test_'
	processes=2
	
	mask_threshold=20
	mask_value=-1
	
	gaussian_blur=0
	gaussian_blur_axis=-1

	param_file=[]

	tofts_parameter_calculator.calc_DCE_properties_single(filepath, T1_tissue, T1_blood, relaxivity, TR, TE, scan_time_seconds, hematocrit, injection_start_time_seconds, flip_angle_degrees, label_file, label_suffix, label_value, mask_value, mask_threshold, T1_map_file, T1_map_suffix, AIF_label_file,  AIF_value_data, AIF_value_suffix, convert_AIF_values, AIF_mode, AIF_label_suffix, AIF_label_value, label_mode, param_file, default_population_AIF, initial_fitting_function_parameters, outputs, outfile_prefix, processes, gaussian_blur, gaussian_blur_axis)

if __name__ == '__main__':

	print 'Entered program..'
	run_test()

	# test('C:/Users/azb22/Documents/Junk/dce_mc_st_corrected.nii')
	# test()
	# tofts_parameter_calculator.test_method_3d()
	# tofts_parameter_calculator.test_method_2d()