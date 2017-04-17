""" A general module for normalization methods. Likely to
	be borrowing heavily from other packages. TODO: Fill in
	these methods.
"""

from ..qtim_utilities.format_util import convert_input_2_numpy

import numpy as np

def zero_mean_unit_variance(input_volume, input_mask=[], output_filename=[]):

    """ Normalization to zero mean and unit variance. Helpful preprocessing for deep learning
        applications. TODO: This function is our first function to take in both numpy and nifti
        files. Maybe make some sort of general function that serves as an interchange between the
        two in the utilities folder. Similar notes for output in nifti or numpy format.
    """

    input_numpy = convert_input_2_numpy(input_volume)

    if input_mask != []:
        mask_numpy = convert_input_2_numpy(input_mask)
    else:
        mask_numpy = []

    if mask_numpy == []:
        vol_mean = np.mean(input_numpy)
        vol_std = np.std(input_numpy)
        output_numpy = (input_numpy - vol_mean) / vol_std        
    else:
        masked_numpy = np.ma.masked_where(mask_numpy == 0, input_numpy)
        vol_mean = np.ma.mean(masked_numpy)
        vol_std = np.ma.std(masked_numpy)
        output_numpy = (masked_numpy - vol_mean) / vol_std
        output_numpy[mask_numpy == 0] = 0

    if output_filename != []:
        save_numpy_2_nifti(output_numpy, normalize_volume, output_filename)
    else:
        return output_numpy


def histogram_normalization(image_numpy, mode='uniform'):

    """ TODO
    """

    return

if __name__ == '__main__':
	pass