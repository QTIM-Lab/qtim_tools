""" A series of functions created to take an ROI from a jpg of a single slice with a red outline, transform that ROI
    into a Nifti file, and extract features from that Nifti file. Requires some manual configuration for finding the
    slice number and orientation of the original JPG image. Requires the package qtim_tools for feature extraction --
    install with "pip install qtim_tools". Requires the program 3D Slicer for registration of different volumes.
"""

import numpy as np
import nibabel as nib
import glob

from ..qtim_utilities.nifti_util import nifti_2_numpy, save_numpy_2_nifti

from subprocess import call
from shutil import copy
from scipy import misc, ndimage

def fill_in_convex_outline(filepath, output_file=[], reference_nifti=[], color_threshold_limits = [[100,300],[0,100],[0,100]] , output_label_num=1,):

    """ Thresholds a jpg according to certain color parameters. Uses a hole-filling algorithm to color in
        regions of interest. TODO: Make work for non-jpgs.
    """

    image_nifti = misc.imread(filepath)
    label_nifti = np.zeros_like(image_nifti)

    red_range = np.logical_and(color_threshold_limits[0][0] < image_nifti[:,:,0], image_nifti[:,:,0] < color_threshold_limits[0][1])
    green_range = np.logical_and(color_threshold_limits[1][0] < image_nifti[:,:,1], image_nifti[:,:,1] < color_threshold_limits[1][1])
    blue_range = np.logical_and(color_threshold_limits[2][0] < image_nifti[:,:,2], image_nifti[:,:,2] < color_threshold_limits[2][1])
    valid_range = np.logical_and(red_range, green_range, blue_range)

    label_nifti[valid_range] = 1
    label_nifti = ndimage.morphology.binary_fill_holes(label_nifti[:,:,0]).astype(label_nifti.dtype)

    # Can save out as either a jpg or a nifti label file.
    if output_file == []:
        return label_nifti
    elif reference_nifti == []:
        misc.imsave(output_file, label_nifti*255)
    else:
        save_numpy_2_nifti(label_nifti, reference_nifti, output_file)

if __name__ == "__main__":

    pass
