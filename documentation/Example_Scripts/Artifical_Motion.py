import numpy as np
import os

from qtim_tools.qtim_utilities.array_util import generate_rotation_affine, save_affine
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

from scipy.io import savemat
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter

def Load_4D_NRRD(input_filepath):

    return convert_input_2_numpy(input_filepath)

def Slicer_Rotate(input_numpy, reference_nifti, affine_matrix, Slicer_path="Slicer"):

    save_nifti_2_numpy(input_numpy, reference_nifti, 'temp.nii.gz')

    Slicer_Command = [Slicer_path, ]

    return


def Generate_Head_Jerk(input_numpy, timepoint, duration, rotation_peaks=[3, 3, 0], reference_nifti=''):

    endpoint = timepoint + duration
    midpoint = np.round(endpoint - timepoint)
    # rotation_matrix_increment = [val for val/(endpoint-midpoint) in rotation_peaks]

    if endpoint > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer than the duration of the volume'

    for t in xrange(input_numpy.shape[-1]):
        if t > timepoint and t < endpoint:
            if t > midpoint:
                current_rot_matrix = -rotation_matrix_increment
            if t <= midpoint:
                current_rot_matrix = rotation_matrix_increment
                Slicer_Rotate(input_numpy[..., timepoint], reference_nifti, current_rot_matrix)

    return


def Generate_Head_Tilt(input_numpy, timepoint, duration):

    if timepoint + duration > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer the the duration of the volume'

def Generate_Deformable_Motion(input_dimensions = (8,8,4), output_dimensions = (256,256,16), output_filepath="Deformable_Matrix", num_matrices = 10):

    zoom_ratio = []
    for i in xrange(len(input_dimensions)):
        zoom_ratio += [output_dimensions[i] // input_dimensions[i]]

    Deformable_Matrix = np.zeros((input_dimensions + (3,)), dtype=float)

    for i in xrange(num_matrices):
        a, b = 4.5, 5.5
        Deformable_Matrix[...,0:2] = (b - a) * np.random.sample(input_dimensions + (2,)) + a

        a, b = .9, 1.1
        Deformable_Matrix[...,2] = (b - a) * np.random.sample(input_dimensions) + a

        Large_Deformable_Matrix = zoom(Deformable_Matrix, zoom_ratio + [1], order=1)
        Large_Deformable_Matrix = gaussian_filter(Large_Deformable_Matrix, sigma=1, order=1)

        print Large_Deformable_Matrix.shape

        output_dict = {}
        output_dict['deformation_matrix_' + str(i)] = Large_Deformable_Matrix

        savemat(output_filepath + '_' + str(i) + '.mat', output_dict)

    return

if __name__ == "__main__":
    Generate_Deformable_Motion()
