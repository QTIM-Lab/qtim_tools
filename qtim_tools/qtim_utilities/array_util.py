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

def pad_image_segment(image, reference_image=''):

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
                image_slice += [slice(slice_num, slice_num+1)]
        else:
            image_slice += [slice(None)]

    return image_numpy[image_slice]

def truncate_image(input_volume, mask_value=0):

    """ This function takes in an N-dimensional array and truncates all rows/columns/etc
        that contain only mask values. Useful for reducing computation time on functions
        whose running time scales exponentially with dimensions size.

        BUG: Currently seems to fail on axes with length 1.
        TODO: Truncate only on some axes.
        TODO: Add the option to add buffer pixels to meet output_dimensions.

        Parameters
        ----------

        input_volume: N-dimensional array
            The volume to be truncated. Will be truncated in every axis.
        mask_value: int or float
            Vectors in an axis that are composed entirely of mask_value will be truncated.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    dims = image_numpy.shape
    truncate_ranges = [[0, 0] for x in dims]

    for axis, axis_length in enumerate(dims):
        start_flag = True
        for idx in range(axis_length):
            if (get_arbitrary_axis_slice(image_numpy, axis, idx) == mask_value).all():
                if start_flag:
                    truncate_ranges[axis][0] = idx + 1
            else:
                start_flag = False
                truncate_ranges[axis][1] = idx + 1

    truncate_slices = [slice(x[0], x[1]) for x in truncate_ranges]

    truncate_image_numpy = image_numpy[truncate_slices]

    return truncate_image_numpy

def truncate_to_maximum_image(input_volume_list, mask_value=0):

    """ Finds the smallest bounding-box which will fit all images.
        For now

        UNIMPLEMENTED
    """

    return

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
        label_indices = np.delete(np.array(label_indices), np.argwhere(label_indices==mask_value))

    for idx in label_indices:
        masked_image = np.copy(image_numpy)
        masked_image[label_numpy != idx] = mask_value
        masked_images += [masked_image]

    return masked_images

def extract_maximal_slice(input_volume, input_label_volume='', mode='max_intensity', axis=2, mask_value=0, return_index=False):

    """ Extracts one slice from a presumably 3D volume. Either take the slice whose label
        has the greatest area (mode='max_label'), or whos sum of voxels has the greatest 
        intensity (mode='max_intensity'), according to the provided axis variable.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    sum_dimensions = tuple([int(x) for x in range(0,image_numpy.ndim) if x != axis])

    if mode == 'max_intensity':
        flattened_image = np.sum(image_numpy, axis=sum_dimensions)
    elif mode == 'max_label':
        label_numpy = convert_input_2_numpy(input_label_volume)
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

    if return_index:
        return get_arbitrary_axis_slice(image_numpy, axis, highest_slice_index), highest_slice_index

    return get_arbitrary_axis_slice(image_numpy, axis, highest_slice_index)

if __name__ == '__main__':
    pass