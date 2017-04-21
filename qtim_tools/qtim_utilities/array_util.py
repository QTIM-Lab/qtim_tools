""" This utility module contains functions meant to do more complex operations
    on arrays. Functions here will frequently be wrapper functions around other
    packages. Numpy is the presumed array standard. 
"""

import numpy as np
import math
from format_util import convert_input_2_numpy
from scipy.ndimage.interpolation import affine_transform, geometric_transform

def get_intensity_range(input_volume, percentiles=[.25,.75]):

    """ Retrieves a min and max of intensities at two specified percentiles on the intensity histogram.
        This could be useful for thresholding, normalizing, or other tasks.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    intensity_range = [np.percentile(image_numpy, percentiles[0], interpolation="nearest"), np.percentile(image_numpy, percentiles[1], interpolation="nearest")]

    return intensity_range

def match_array_orientation(image1, image2):

    """ TODO function. Write some smart way to match flips and
        90 degree rotations between ararys with different orientations
        This may require the existence of a header construct.
    """
    return

def pad_image_segment(image, reference_image):

    """ Many people store their label maps in Niftis with dimensions smaller
        than the corresponding image. This is also that natural output of DICOM-SEG
        nifti conversions. Padding these arrays with empty values so that they are
        comparable requires knowledge of that image's origin. TODO: write a function
        that can do this with some specifications from either a reference image or
        header.

        It may be better to put a version of this function specifically for DICOMSEG
        into the dicom_util module.
    """

    return

def get_arbitrary_axis_slice(input_volume, axis, slice_num):

    """ Returns a slice of numpy array according to an arbitrary axis. This is a
        convencience function mostly because I find Python's slice notation a bit
        cumbersome.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    image_slice = []

    for dim in xrange(image_numpy.ndim):
        if dim == axis:
            if slice_num is list:
                image_slice += [slice(slice_num[0], slice_num[1])]
            else:
                image_slice += [slice(slice_num)]
        else:
            image_slice += [slice(None)]

    return image_numpy[image_slice]

def truncate_image(input_volume, mask_value=0):

    """ There are better ways online to do what I am attempting,
        but so far I have not gotten any of them to work. In the meantime,
        this long and probably ineffecient code will suffice. It is
        meant to remove empty rows from images. Currently only works with
        3D images.

        TODO: Condense code.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    dims = image_numpy.shape
    truncate_ranges = [[0, x] for x in dims]

    for dim in enumerate(dims):
        start_flag = True
        for idx in xrange(dim):
            if (get_arbitrary_axis_slice(image_numpy, dim, idx) == mask_value).all():
                if start_flag:
                    truncate_ranges[dim][0] = idx + 1
            else:
                start_flag = False
                truncate_ranges[dim][0] = idx + 1

    truncate_slices = [slice(x[0], x[1]) for x in truncate_ranges]

    truncate_image_numpy = image_numpy[truncate_slices]

    return truncate_image_numpy

def split_image(input_volume, input_label_volume='', label_indices='', mask_value=0):

    """ This function takes in an image, optionally a label image, and optionally a set of indices,
        and returns one duplicate masked image for each given label. Useful for analyzing,
        say, multiple tumors, although expensive in memory. Useful when paired with the
        truncate_image function to reduce array memory.
    """

    image_numpy = convert_input_2_numpy(input_volume)
    label_numpy = convert_input_2_numpy(input_label_volume)

    masked_images = []

    if label_indices == '':
        if label_numpy == '':
            label_indices = np.unique(image_numpy)
        else:
            label_indices = np.unique(label_numpy)
        if mask_value in label_indices:
            label_indices.remove(mask_value)

    for idx in label_indices:
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = mask_value
        masked_images += [masked_image]

    return masked_images

def extract_maximal_slice(input_volume, input_label_volume='', mode='max_intensity', axis=2, mask_value=0):

    """ Extracts one slice from a presumably 3D volume. Either take the slice whose label
        has the greatest area (mode='max_label'), or whos sum of voxels has the greatest 
        intensity (mode='max_intensity'), according to the provided axis variable.
    """

    image_numpy = convert_input_2_numpy(input_volume)
    if input_label_volume != '':
        label_numpy = convert_input_2_numpy(input_label_volume)

    sum_dimensions = range(0,image_numpy.ndim).pop(axis)

    if mode == 'max_intensity':
        flattened_image = np.sum(image_numpy, axis=sum_dimensions)
    elif mode == 'max_label':
        flattened_image = np.sum(label_numpy, axis=sum_dimensions)
    elif mode == 'non_mask':
        flattened_image = (image_numpy != mask_value).sum(axis=sum_dimensions)
    else:
        print 'Invalid mode entered to extract_maximal_slice_3d. Returning original array..'
        return image_numpy

    # TODO: Put in support for 
    highest_slice_index = np.argmax(flattened_image)
    try:
        highest_slice_index = highest_slice_index[0]
    except:
        pass

    return get_arbitrary_axis_slice(image_numpy, axis, highest_slice_index)

def generate_identity_affine():

    """ A convenient function for generating an identity affine matrix. Can be
        used for saving blank niftis.
    """

    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

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
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[translation_distance,0,0,1]]
    
    elif axis == 1:
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,translation_distance,0,1]]
    
    elif axis == 2:
        return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,translation_distance,1]]
    
    else:
        print 'Error, can only accept axes 0-2 as input to axis parameter.'
        return []

def apply_affine(input_volume, affine_matrix, method="python"):

    """ Provides methods for applying an affine matrix to a 3D volume. TODO:
        extend this past 3D volumes. Also has a method to apply the matrix in
        Slicer, if Slice is available. Slicer will be much faster, but requires
        a special array format.
    """

    if method == 'python':

        input_numpy = convert_input_2_numpy(input_volume)

        def affine_calculation(output_coords):
            output_coords = output_coords + (0,)
            return tuple(np.matmul(affine_matrix, np.array(output_coords))[:-1])

        return geometric_transform(input_numpy, affine_calculation)

    elif method == 'slicer':
        pass

    else:
        print 'Invalid method parameter. Returning []'
        return []

def save_affine(affine_matrix, output_filename, output_format="itk_affine"):

    """ Saves a numpy affine matrix to ITK format for use in other programs,
        e.g. 3D Slicer. This method is not complete - there is a lot of mapping
        to do from 
    """

    if output_format == "itk_affine":
        f = open(output_filename, 'w')
        f.write('#Insight Transform File V1.0')
        f.write('Transform: AffineTransform_double_3_3')
# Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
# FixedParameters: 0 0 0

    else:
        print 'Invalid output format. Returning []'
        return []

def get_jacobian_determinant(input_volume):

    input_numpy = convert_input_2_numpy(input_volume)

    jacobian_output = np.zeros_like(input_numpy)

    temp_jacobian = np.zeros((input_numpy.shape[0:-1] + (input_numpy.shape[-1],input_numpy.shape[-1])), dtype=float)


    for r in xrange(input_numpy.shape[-1]):
        for c in xrange(input_numpy.shape[-1]):
            temp_jacobian[...,r,c] = np.gradient(input_numpy[..., c])[r]

    return np.linalg.det(temp_jacobian)

if __name__ == '__main__':
    pass