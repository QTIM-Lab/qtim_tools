import numpy as np
import math
import os

from format_util import convert_input_2_numpy
from scipy.ndimage.interpolation import affine_transform, geometric_transform
from subprocess import call

def generate_identity_affine(timepoints=1):

    """ A convenient function for generating an identity affine matrix. Can be
        used for saving blank niftis.
    """

    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]*timepoints)

def generate_rotation_affine(axis=0, rotation_degrees=1):

    """ This function creates an affine transformation matrix with a set rotation at a set axis.
        Code ripped from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/. Needs
        added functionality to set a center point. There is a method available in OpenCV to
        do this, but for now I am holding off on making OpenCV a requirement for qtim_tools.
    """
     
    rotation_radians = math.radians(rotation_degrees)

    if axis == 0:
        R = np.array([[1,0,0, 0],
                        [0,math.cos(rotation_radians), -math.sin(rotation_radians), 0 ],
                        [0,math.sin(rotation_radians), math.cos(rotation_radians), 0  ],
                        [0,0,0,1]])  
    
    elif axis == 1:  
        R = np.array([[math.cos(rotation_radians),0,math.sin(rotation_radians), 0],
                    [0,1,0, 0],
                    [-math.sin(rotation_radians),0,math.cos(rotation_radians)  , 0],
                    [0,0,0,1]])
    
    elif axis == 2:                 
        R = np.array([[math.cos(rotation_radians),-math.sin(rotation_radians),0, 0],
                    [math.sin(rotation_radians),math.cos(rotation_radians),0, 0],
                    [0,0,1, 0],
                    [0,0,0,1]])

    else:
        print 'Error, can only accept axes 0-2 as input to axis parameter.'
        return []
 
    return R

def generate_translation_affine(axis=0, translation_distance=10):

    """ This function creates an affine transformation matrix with a set translation at a set axis.
        Code ripped from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/. Needs
        added functionality to set a center point. There is a method available in OpenCV to
        do this, but for now I am holding off on making OpenCV a requirement for qtim_tools.
    """

    if axis == 0:
        T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[translation_distance,0,0,1]]
    
    elif axis == 1:
        T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,translation_distance,0,1]]
    
    elif axis == 2:
        T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,translation_distance,1]]
    
    else:
        print 'Error, can only accept axes 0-2 as input to axis parameter.'
        return []

    return np.array(T)

def apply_affine(input_volume, affine_matrix, method="python", Slicer_path="Slicer", reference_file=''):

    """ Provides methods for applying an affine matrix to a 3D volume. TODO:
        extend this past 3D volumes. Also has a method to apply the matrix in
        Slicer, if Slice is available. Slicer will be much faster, but requires
        a special array format.

        TODO: Get this working for 4D Volumes.
    """

    input_numpy = convert_input_2_numpy(input_volume)

    if method == 'python':

        def affine_calculation(output_coords):
            output_coords = output_coords + (0,)
            return tuple(np.matmul(affine_matrix, np.array(output_coords))[:-1])

        return geometric_transform(input_numpy, affine_calculation)

    elif method == 'slicer':

        save_numpy_2_nifti(input_numpy, reference_nifti, 'temp.nii.gz')
        save_affine(affine_matrix, 'temp.txt')

        Slicer_Command = [Slicer_path, '--launch', 'ResampleScalarVectorDWIVolume', 'temp.nii.gz', 'temp_out.nii.gz', '-f', 'temp.txt', '-i', 'bs']

        call(' '.join(Slicer_Command), shell=True)

        output_array = convert_input_2_numpy('temp_out.nii.gz')
        
        os.remove('temp.nii.gz')
        os.remove('temp.txt')
        os.remove('temp_out.nii.gz')

        return output_array

    # return convert_input_2_numpy('temp_out.nii.gz')

    else:
        print 'Invalid method parameter. Returning []'
        return []

def compose_affines(affine_matrix_1, affine_matrix_2):

    """ Simple matrix multiplication. For time series, matrix multiplication
        is component-wise. Questionable if there are use cased for 4D+ dimnesions
        but those currently do not work.
    """

    if affine_matrix_1.ndim == 2:

        return np.matmul(affine_matrix_1, affine_matrix_2)

    elif affine_matrix_1.ndim == 3:

        output_matrix = np.zeros_like(affine_matrix_1)

        for t in xrange(affine_matrix_1.shape[-1]):
            output_matrix[..., t] = np.matmul(affine_matrix_1[..., t], affine_matrix_2[...,t])

        return output_matrix

    else:
        print 'Error: input matrix has incorrect number of dimensions (4x4xN) for comptuation'
        return []

def save_affine(affine_matrix, output_filename, output_format="itk_affine"):

    """ Saves a numpy affine matrix to ITK format for use in other programs,
        e.g. 3D Slicer. This method is not complete - there is a lot of mapping
        to do from data types to transform types.
    """

    if output_format == "itk_affine":
        f = open(output_filename, 'w')
        f.write('#Insight Transform File V1.0 \n')
        f.write('Transform: AffineTransform_double_3_3 \n')
        
        rotate_string = ''
        translate_string = ''

        for row in affine_matrix:
            rotate_string = rotate_string + " ".join(map(str, row[:-1])) + ' '
            translate_string = translate_string + str(row[-1]) + ' '
        translate_string = translate_string[0:-2]

        f.write('Parameters: ' + rotate_string + '\n')
        f.write('FixedParameters: ' + translate_string + '\n')

    else:
        print 'Invalid output format. Returning []'
        return []


def generate_motion_jerk(duration, timepoint=0, rotation_peaks=[3, 3, 0], total_timepoints=-1, input_motion_array=[]):

    """ Generates an affine jerk that smoothly moves between an affine displacement and back again.
        TODO: Allow custom ordering of rotations. Currently all rotations go first by x, then by y,
        then by z, etc.
    """

    if input_motion_array != []:
        total_timepoints = input_motion_array.shape[-1]

    if total_timepoints == -1:
        total_timepoints = duration

    endpoint = timepoint + duration
    midpoint = timepoint + np.round(endpoint - timepoint)/2
    rotation_matrix_increment = np.array([float(x)/float(timepoint-endpoint) for x in rotation_peaks])

    print timepoint, endpoint, duration, total_timepoints

    if endpoint > total_timepoints:
        print 'Invalid timepoint, longer than the duration of the volume'

    rotation_direction = np.array([0,0,0])

    output_motion_array = np.zeros((4,4,total_timepoints), dtype=float)

    for t in xrange(total_timepoints):

        current_rotation_matrix = generate_identity_affine()

        if t > timepoint and t < endpoint:
            
            if t > midpoint:
                rotation_direction = rotation_direction - rotation_matrix_increment
            if t <= midpoint:
                rotation_direction = rotation_direction + rotation_matrix_increment

            for axis, value in enumerate(rotation_direction):
                current_rotation_matrix = np.matmul(current_rotation_matrix, generate_rotation_affine(axis, value))

        if input_motion_array != []:
            output_motion_array[..., t] = np.matmul(input_motion_array, current_rotation_matrix)
        else:
            output_motion_array[..., t] = current_rotation_matrix

    return output_motion_array

def generate_motion_tilt(timepoint, duration, rotation_peaks=[3, 3, 0], input_filepath='', reference_nifti='', output_filepath=''):

    """ Generates an affine jerk that smoothly moves between an affine displacement, and then stays there. 
        TODO: Allow custom ordering of rotations. Currently all rotations go first by x, then by y,
        then by z, etc.
    """

    if input_motion_array != '':
        total_timepoints = input_motion_array.shape[-1]

    if total_timepoints == -1:
        total_timepoints = duration

    endpoint = timepoint + duration
    midpoint = timepoint + np.round(endpoint - timepoint)/2
    rotation_matrix_increment = np.array([float(x)/float(timepoint-endpoint) for x in rotation_peaks])

    if endpoint > total_timepoints:
        print 'Invalid timepoint, longer than the duration of the volume'

    rotation_direction = np.array([0,0,0])

    output_motion_array = np.zeros((4,4,total_timepoints), dtype=float)

    for t in xrange(total_timepoints):

        current_rotation_matrix = generate_identity_affine()

        if t > timepoint and t < endpoint:
            
            if t > midpoint:
                rotation_direction = rotation_direction - rotation_matrix_increment
            if t <= midpoint:
                rotation_direction = rotation_direction + rotation_matrix_increment

            for axis, value in enumerate(rotation_direction):
                current_rotation_matrix = np.matmul(current_rotation_matrix, generate_rotation_affine(axis, value))

        if input_motion_array != []:
            output_motion_array[..., t] = np.matmul(input_motion_array, current_rotation_matrix)
        else:
            output_motion_array[..., t] = current_rotation_matrix

    return output_motion_array

def generate_noisy_motion():

    return

def get_jacobian_determinant(input_volume):

    """ Takes in an vector-valued space, calculates that spaces gradients and jacobian matrices,
        and returns a scalar-valued space with the jacobian determinants.

        TODO: Does not seem to work in the 1D case, where jacobian determinant is equal to the
        gradient.
    """

    input_numpy = convert_input_2_numpy(input_volume)

    jacobian_output = np.zeros_like(input_numpy)

    temp_jacobian = np.zeros((input_numpy.shape[0:-1] + (input_numpy.shape[-1],input_numpy.shape[-1])), dtype=float)

    for r in xrange(input_numpy.shape[-1]):
        for c in xrange(input_numpy.shape[-1]):
            temp_jacobian[...,r,c] = np.gradient(input_numpy[..., c])[r]

    return np.linalg.det(temp_jacobian)

def return_jacobian_matrix(input_volume, index):

    """ Returns a Jacobian matrix for a certain index. This function is currently broken and not very
        generalizable. TODO: fix.
    """

    input_numpy = convert_input_2_numpy(input_volume)

    jacobian_output = np.zeros_like(input_numpy)

    temp_jacobian = np.zeros((input_numpy.shape[0:-1] + (input_numpy.shape[-1],input_numpy.shape[-1])), dtype=float)

    for r in xrange(input_numpy.shape[-1]):
        for c in xrange(input_numpy.shape[-1]):
            temp_jacobian[...,r,c] = np.gradient(input_numpy[..., c])[r]

    return temp_jacobian[index[0],index[1],index[2], :,:]