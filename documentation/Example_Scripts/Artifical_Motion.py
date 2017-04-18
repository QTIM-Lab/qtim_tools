import numpy as np

from qtim_tools.qtim_utilities.array_util import generate_rotation_matrix
from qtim_tools.format_util import convert_input_2_numpy

def Load_4D_NRRD(input_filepath):

    return convert_input_2_numpy(input_filepath)

def Save_Slice(input_numpy, reference_nifti, slice_num):

    return

def Slicer_Rotate(input_filename, affine_matrix):

    return


def Generate_Head_Jerk(input_numpy, timepoint, duration, rotation_peaks=[3, 3, 0]):

    endpoint = timepoint + duration
    midpoint = np.round(endpoint - timepoint)

    if endpoint > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer than the duration of the volume'

    for t in xrange(input_numpy.shape[-1]):
        if t > timepoint and t < endpoint:
            if t > midpoint:
                for axis_idx, rotation in enumerate(rotation_peaks):
                    generate_rotation_matrix(axis_idx, rotation*(endpoint-midpoint))


def Generate_Head_Tilt(input_numpy, timepoint, duration):

    if timepoint + duration > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer the the duration of the volume'

def Generate_Deformable_Motion(input_numpy):

    return



