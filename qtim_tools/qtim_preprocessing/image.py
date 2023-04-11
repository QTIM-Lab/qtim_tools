""" This preprocessing module is a little vague. As currently construed, it is meant to
    work on images received in .PNG, .JPG format, etc.
"""

import numpy as np
import nibabel as nib
import glob

from subprocess import call
from shutil import copy
from scipy import misc, ndimage
from skimage import measure

from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy, save_numpy_2_nifti
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

def fill_in_convex_outline(input_data, output_file=None, reference_nifti=None, threshold_limit=[0,100], color_threshold_limits= [[100,300],[0,100],[0,100]], output_label_num=1):

    """ Thresholds a jpg according to certain color parameters. Uses a hole-filling algorithm to color in
        regions of interest.

        TODO: Reorganize into two separate tracks, instead of two winding tracks.
    """

    image_nifti, image_type = convert_input_2_numpy(input_data, return_type=True)

    label_nifti = np.zeros_like(image_nifti)

    if image_type == 'image':

        red_range = np.logical_and(color_threshold_limits[0][0] < image_nifti[:,:,0], image_nifti[:,:,0] < color_threshold_limits[0][1])
        green_range = np.logical_and(color_threshold_limits[1][0] < image_nifti[:,:,1], image_nifti[:,:,1] < color_threshold_limits[1][1])
        blue_range = np.logical_and(color_threshold_limits[2][0] < image_nifti[:,:,2], image_nifti[:,:,2] < color_threshold_limits[2][1])
        valid_range = np.logical_and(red_range, green_range, blue_range)

        label_nifti[valid_range] = 1

        label_nifti = ndimage.morphology.binary_fill_holes(label_nifti[:,:,0]).astype(label_nifti.dtype)
    
        if output_file is not None:
            misc.imsave(output_file, label_nifti*255)

    else:
        image_nifti[image_nifti != 0] = output_label_num
        if image_nifti.ndim == 3:
            for z in range(image_nifti.shape[2]):
                label_nifti[..., z] = ndimage.morphology.binary_fill_holes(image_nifti[...,z]).astype(image_nifti.dtype)
        else:
            label_nifti = ndimage.morphology.binary_fill_holes(image_nifti).astype(image_nifti.dtype)

        print(np.sum(label_nifti), 'HOLE FILLED SUM')

        if output_file is not None:
            save_numpy_2_nifti(label_nifti, reference_nifti, output_file)

    return label_nifti

def split_islands():
    pass

if __name__ == "__main__":

    pass
