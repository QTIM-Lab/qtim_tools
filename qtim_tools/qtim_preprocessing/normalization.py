""" A general module for normalization methods. Likely to
	be borrowing heavily from other packages. TODO: Fill in
	these methods.
"""

from ..qtim_utilities.format_util import convert_input_2_numpy

import numpy as np

def zero_mean_unit_variance(input_volume, input_mask=[], output_filename=[]):

    """ Normalizes an image by subtracting its mean and dividing by its standard
        deviation. If provided a mask, the normalization will only occur within the
        mask.

        TODO: Add support for not-equal-to masking.
        TODO: Add support for other replacement_values, a-la scikit-learn.

        Parameters
        ----------

        input_data: N-dimensional array or str
            The volume to be normalized. Can be filename or numpy array.
        input_mask: N-dimensional array or str
            A label mask. Must be the same size as input_data
        output_filename: str
            Optional, save output to filename.

        Returns
        -------
        input_numpy: array
            Transformed input array.
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