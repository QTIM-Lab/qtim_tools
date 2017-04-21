import numpy as np
import os

from qtim_tools.qtim_utilities.array_util import generate_rotation_affine, save_affine, get_jacobian_determinant
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

def Add_White_Noise(input_filepath, output_filepath, noise_scale=1, noise_multiplier=10):

    input_numpy = convert_input_2_numpy(input_filepath)

    for t in xrange(input_numpy.shape[-1]):
        input_numpy[..., t] = input_numpy[..., t] + np.random.normal(scale=noise_scale, size=input_numpy[..., t].shape).reshape(input_numpy[..., t].shape) * noise_multiplier

    save_numpy_2_nifti(input_numpy, input_filepath, output_filepath)

def Generate_Deformable_Motion(input_dimensions = (8,8,4), output_dimensions = (256,256,16), output_filepath="/home/abeers/Projects/DCE_Motion_Phantom/Deformable_Matrix", num_matrices = 65, deformation_scale=1):

    zoom_ratio = []
    for i in xrange(len(input_dimensions)):
        zoom_ratio += [output_dimensions[i] // input_dimensions[i]]

    Deformable_Matrix = np.zeros((input_dimensions + (3,)), dtype=float)

    output_dict = {}

    Final_Deformation_Matrix = np.zeros((256,256,16,3,65), dtype=float)

    for i in xrange(num_matrices):
        a, b = -5*deformation_scale, 5*deformation_scale
        Deformable_Matrix[...,0:2] = (b - a) * np.random.sample(input_dimensions + (2,)) + a

        c, d = -1*deformation_scale, 1*deformation_scale
        Deformable_Matrix[...,2] = (d - c) * np.random.sample(input_dimensions) + c

        Jacobian_Matrix = get_jacobian_determinant(Deformable_Matrix)
        print (Jacobian_Matrix < 0).sum()
        print (Jacobian_Matrix >= 0).sum()

        while (Jacobian_Matrix < 0).sum() > 0:
            for count, index in enumerate(np.ndindex(Jacobian_Matrix.shape)):
                while Jacobian_Matrix[index] < 0 and index[0] > 0 and index[1] > 0 and index[2] > 0 and index[0] < 7 and index[1] < 7 and index[2] < 3:
                    # print 'current count', count
                    # print index
                    Deformable_Matrix[index[0]-1:index[0]+2, index[1]-1:index[1]+2, index[2]-1:index[2]+2] = [(b - a) * np.random.sample((3,3,3)) + a, (b - a) * np.random.sample((3,3,3)) + a, (d - c) * np.random.sample((3,3,3)) + c]
                    # print Deformable_Matrix[index]
                    Jacobian_Matrix = get_jacobian_determinant(Deformable_Matrix)
                    print Jacobian_Matrix[index]
            print (Jacobian_Matrix < 0).sum()

        # for d in xrange(Deformable_Matrix.shape[-1]):
            # print 'Gradient ', d
            # print np.gradient(Deformable_Matrix[d])[0]

        # Large_Deformable_Matrix = zoom(Deformable_Matrix, zoom_ratio + [1], order=1)

        # Large_Deformable_Matrix[...,0:2] = gaussian_filter(Large_Deformable_Matrix[...,0:2], sigma=1)
        # Large_Deformable_Matrix[...,2] = gaussian_filter(Large_Deformable_Matrix[...,2], sigma=1)

        # Final_Deformation_Matrix[...,i] = Large_Deformable_Matrix

    # output_dict['deformation_matrix'] = Final_Deformation_Matrix

    # savemat(output_filepath, output_dict)

    return

if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    Generate_Deformable_Motion(num_matrices=1)

    # for noise_types in [['low', 5],['mid', 10],['high', 20]]:
        # Add_White_Noise(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', noise_multiplier=noise_types[1])
    # for noise_types in [['lowest', .25],['low', .5],['mid', 1],['high', 2]]:
        # Generate_Deformable_Motion(output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/Deformable_Matrix_' + noise_types[0],  deformation_scale=noise_types[1])
