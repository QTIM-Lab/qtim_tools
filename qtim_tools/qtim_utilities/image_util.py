""" Utilities for dealing with 2D image files, like JPG, PNG,
    etc. Most functions will likely be wrappers around other
    libraries, e.g. PIL, skimage, etc. Consider merging the
    image library from qtim_preprocessing.
"""

import nibabel as nib
import numpy as np
import scipy.misc

def img_2_numpy(input_image):
    
    """ Loads image data and returns a numpy array. There
        will likely be many parameters to specify because
        of the strange quantization issues seemingly inherent
        in loading images.
    """

    image_nifti = misc.imread(filepath)

    return image_nifti

def save_numpy_2_img(input_numpy, output_filename, output_dimensions=[], rescale=0):

    if output_dimensions:
        input_numpy = scipy.misc.imresize(input_numpy, output_dimensions)

    elif rescale > 0:
        input_numpy = scipy.mis.imresize(input_numpy, rescale)

    scipy.misc.imsave(output_filename, input_numpy)