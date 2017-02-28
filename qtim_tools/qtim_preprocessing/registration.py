""" This is a wrapper script for BRAINSFit registration by 3D Slicer. In the future, there could be an all-Python
	implementation of registration in this function. In the meantime, one will need 3D Slicer (or a Docker container
	with 3DSlicer inside).
"""

from qtim_tools.qtim_utilities import nifti_util
import numpy as np
# from PIL import Image
import glob
from subprocess import call

def register_to_one(filepath, Slicer_Path, registration_folder, output_suffix = '', output_folder='', file_regex='*.nii*',transform_type='Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine', transform_mode = 'useMomentsAlign', interpolation_mode = 'Linear', sampling_percentage = .06):

	if output_folder == '':
		output_folder = registration_folder

	BRAINSFit_base_command = [Slicer_Path,'--launch', 'BRAINSFit', '--fixedVolume', '"' + filepath + '"', '--transformType', transform_type, '--initializeTransformMode', transform_mode, '--interpolationMode', interpolation_mode, '--samplingPercentage', str(sampling_percentage)]

	moving_volumes = glob.glob(registration_folder + file_regex)

	for moving_volume in moving_volumes:

		no_path = str.split(moving_volume ,'\\')
		file_prefix = str.split(no_path[-1], '.nii')[0]

		if '.nii.gz' in moving_volume:
			output_filename = output_folder + file_prefix + output_suffix + '.nii.gz'
		else:
			output_filename = output_folder + file_prefix + output_suffix + '.nii'

		BRAINSFit_specific_command = BRAINSFit_base_command + ['--movingVolume','"' + no_path[0] +  '/' + no_path[1] + '"','--outputVolume','"' + output_filename + '"']

		# print ' '.join(BRAINSFit_specific_command)

		try:
			# call(' '.join(BRAINSFit_specific_command), shell=True)
			print 'cp ' + '"' + no_path[0] +  '/' + no_path[1] + '" "' + output_folder + file_prefix + '_r_T2-label' + '.nii.gz"'
			call('cp ' + '"' + no_path[0] +  '/' + no_path[1] + '" "' + output_folder + file_prefix + '_r_T2-label' + '.nii.gz"')
		except:
			pass

	return

def run_test():

	Slicer_Path = '"C:/Users/azb22/Documents/Software/SlicerNightly/Slicer 4.6.0/Slicer.exe"'
	fixed_volume = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/7_Ax_T2_PROPELLER.nii.gz'
	moving_folder = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/'
	output_folder = 'C:/Users/azb22/Documents/Scripting/Tata_Hospital/Drawn_ROI_TestFiles/Registered_Volumes/'

	register_to_one(fixed_volume, Slicer_Path, moving_folder, '_r_T2',output_folder)

	return

if __name__ == "__main__":
	run_test()