""" This program is meant to generate phantom data for testing purposes.
"""

import numpy as np
import nibabel as nib
import os
from qtim_tools.qtim_utilities import nifti_util


def size_cube_phantom(reference_image, output_folder):

    """ Takes an input image and outputs
    """

    nifti_3d = nib.load(reference_image)
    image_3d = nifti_3d.get_fdata()

    for size_ratio in np.arange(.1, 1, .1):
        phantom_3d = np.zeros((image_3d.shape[0], image_3d.shape[1], 1))
        phantom_3d[(phantom_3d.shape[0]*(size_ratio/2)):(phantom_3d.shape[0] - (phantom_3d.shape[0]*(size_ratio/2))), (phantom_3d.shape[1]*(size_ratio/2)):(phantom_3d.shape[1] - (phantom_3d.shape[1]*(size_ratio/2))),0] = 1
        nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'Size_' + str(int(10*(1-size_ratio))) + '_Phantom.nii.gz'))
        nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'Size_' + str(int(10*(1-size_ratio))) + '_Phantom-label.nii.gz'))


def glcm_cube_phantom(reference_image, output_folder):

    nifti_3d = nib.load(reference_image)
    image_3d = nifti_3d.get_fdata()

    for direction in ['Vertical', 'Horizontal', 'Grid']:
        for alternation_rate in np.arange(0, 6, 1):
            phantom_3d = 100 + 10 * np.random.randn(200, 200, 2)
            label_3d = np.zeros_like(phantom_3d)
            label_3d[0:20, 0:20,:] = 1
            phantom_3d[phantom_3d < 0] = 0

            indice_list = []
            both_indice_list = []

            for i in range(200):
                if i % (alternation_rate*2) < alternation_rate:
                    indice_list += [i]
                    # both_indice_list ++ [i, i, 1]
                    # both_indice_list ++ [i, i, 2]

            if alternation_rate == 0:
                pass
            elif direction == 'Vertical':
                phantom_3d[indice_list, :, :] += 100
            elif direction == 'Horizontal':
                phantom_3d[:, indice_list, :] += 100
            elif direction == 'Grid':
                for indice in indice_list:
                    phantom_3d[indice, indice_list, :] += 100

            nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'GLCM_' + direction + '_' + str(alternation_rate) + '_Phantom.nii.gz'))
            nifti_util.save_numpy_2_nifti(label_3d, reference_image, os.path.join(output_folder, 'GLCM_' + direction + '_' + str(alternation_rate) + '_Phantom-label.nii.gz'))

            print([direction, alternation_rate])

def intensity_cube_phantom(reference_image, output_folder):

    nifti_3d = nib.load(reference_image)
    image_3d = nifti_3d.get_fdata()

    for phantom_type in ['grey','split','checker','noisy_grey','one_spot']:
        phantom_3d = np.zeros((200, 200, 2))
        label_3d = np.zeros_like(phantom_3d)
        label_3d[70:130, 70:130,:] = 1
        phantom_3d[phantom_3d < 0] = 0

        if phantom_type == 'grey':
            phantom_3d[:,:,:] = 100

        elif phantom_type == 'split':
            phantom_3d[0:100,:,:] = 50
            phantom_3d[100:,:,:] = 150

        elif phantom_type == 'checker':
            indice_list = []
            for i in range(200):
                if i % (20) < 10:
                    indice_list += [i]
            phantom_3d[:,:,:] = 50
            for indice in indice_list:
                phantom_3d[indice, indice_list, :] = 150

        elif phantom_type == 'noisy_grey':
            phantom_3d = 100 + 10 * np.random.randn(200, 200, 2)
            phantom_3d[phantom_3d < 0] = 0

        elif phantom_type == 'one_spot':
            phantom_3d[:,:,:] = 100
            phantom_3d[80:100, 80:100, :] = 150

        nifti_util.save_numpy_2_nifti(phantom_3d, reference_image, os.path.join(output_folder, 'Intensity_' + phantom_type + '_Phantom.nii.gz'))
        nifti_util.save_numpy_2_nifti(label_3d, reference_image, os.path.join(output_folder, 'Intensity_' + phantom_type + '_Phantom-label.nii.gz'))

        print([phantom_type])

    return

def get_phantom_filepath(input):

    if input == 'intensity_square':
        return os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Intensity'))
    elif input == 'glcm_square':
        return os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_GLCM'))
    elif input == 'size_square':
        return os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Size'))
    elif input == 'size_mri':
        return os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','MR_Tumor_Size'))
    elif input == 'shape_mri':
        return os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','MR_Tumor_Shape'))
    else:
        print('Sorry, there is no available phantom data by that keyword. Available keywords: "intensity_square", "glcm_square", "size_square", "size_mri", "shape_mri".')
        return ''

def test_method():
    return

if __name__ == '__main__':
    # test_method()

    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','MR_Tumor_Size','MR_BrainTumor_SizeTest.nii.gz'))
    test_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','test_data_features','Phantom_Intensity'))

    intensity_cube_phantom(test_file, test_folder)
