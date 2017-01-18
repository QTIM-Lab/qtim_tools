import numpy as np
import nibabel as nib
import os
from qtim_tools.qtim_utilities import nifti_util

def size_cube_phantom(reference_image, output_folder):

	""" Takes an input image and outputs
	"""

	nifti_3d = nib.load(reference_image)
	image_3d = nifti_3d.get_data()

	for size_ratio in np.arange(.1, 1, .1):
		phantom_3d = np.zeros((image_3d.shape[0], image_3d.shape[1], 1))
		phantom_3d[(phantom_3d.shape[0]*(size_ratio/2)):(phantom_3d.shape[0] - (phantom_3d.shape[0]*(size_ratio/2))), (phantom_3d.shape[1]*(size_ratio/2)):(phantom_3d.shape[1] - (phantom_3d.shape[1]*(size_ratio/2))),0] = 1
		nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'Size_' + str(int(10*(1-size_ratio))) + '_Phantom.nii.gz'))
		nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'Size_' + str(int(10*(1-size_ratio))) + '_Phantom-label.nii.gz'))

# def glcm_cube_phantom(reference_image, output_folder):

# 	nifti_3d = nib.load(reference_image)
# 	image_3d = nifti_3d.get_data()

# 	for direction in ['vert', 'horz']:
# 		for alternation_rate in np.arange(1, 6, 1):
# 			phantom_3d = np.zeros((image_3d.shape[0], image_3d.shape[1], 1))


def test_method():
	return

if __name__ == '__main__':
	# test_method()

	test_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','MR_Tumor_Size','MR_BrainTumor_SizeTest.nii.gz'))
	test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Size'))

	size_cube_phantom(test_file, test_folder)