import tofts_parameter_calculator
# from ..qtim_utilities import nifti_util

def test(filepath=[]):
	tofts_parameter_calculator.test_method_3d(filepath)

if __name__ == '__main__':
	test('C:/Users/azb22/Documents/Junk/dce_mc_st_corrected.nii')
	# tofts_parameter_calculator.test_method_3d()