import numpy as np
import os

from qtim_tools.qtim_utilities.array_util import generate_rotation_matrix, save_affine
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_nifti_2_numpy

def Load_4D_NRRD(input_filepath):

    return convert_input_2_numpy(input_filepath)

def Slicer_Rotate(input_numpy, reference_nifti, affine_matrix):

    save_nifti_2_numpy(input_numpy, reference_nifti, 'temp.nii.gz')

    Slicer_Command = 

    return


def Generate_Head_Jerk(input_numpy, timepoint, duration, rotation_peaks=[3, 3, 0], reference_nifti):

    endpoint = timepoint + duration
    midpoint = np.round(endpoint - timepoint)
    rotation_matrix_increment = [val for val/(endpoint-midpoint) in rotation_peaks]

    if endpoint > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer than the duration of the volume'

    for t in xrange(input_numpy.shape[-1]):
        if t > timepoint and t < endpoint:
            if t > midpoint:
                current_rot_matrix = -rotation_matrix_increment
            if t <= midpoint:
                current_rot_matrix = rotation_matrix_increment
                Slicer_Rotate(input_numpy[..., timepoint], reference_nifti, current_rot_matrix)



def Generate_Head_Tilt(input_numpy, timepoint, duration):

    if timepoint + duration > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer the the duration of the volume'

def Generate_Deformable_Motion(input_numpy):

    return

if __name__ == "__main__":
