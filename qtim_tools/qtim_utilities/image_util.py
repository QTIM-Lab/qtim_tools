""" Utilities for dealing with 2D image files, like JPG, PNG,
    etc. Most functions will likely be wrappers around other
    libraries, e.g. PIL, skimage, etc. Consider merging the
    image library from qtim_preprocessing.
"""

import nibabel as nib
import numpy as np
import nrrd
from nifti_util import nifti_2_numpy, save_numpy_2_nifti

def img_2_numpy(input_image):
    
    """ Loads image data and returns a numpy array. There
        will likely be many parameters to specify because
        of the strange quantization issues seemingly inherent
        in loading images.
    """

    return