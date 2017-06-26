""" This module should be used for functions that null out values in an array
    based on a condition. Primarily used for masking.
"""

import numpy as np

from ..qtim_utilities.format_util import convert_input_2_numpy
from ..qtim_utilities.nifti_util import save_numpy_2_nifti

def crop_with_mask(input_data, label_data, output_filename='', mask_value=0, return_labels=None, replacement_value=0):

    """ Crops and image with a predefined mask image. Values equal to mask value
        are replaced with replacement_value.

        TODO: Add support for not-equal-to masking.
        TODO: Add support for other replacement_values, a-la scikit-learn.
        TODO: Add support for multiple return_labels.

        Parameters
        ----------

        input_data: N-dimensional array or str
            The volume to be cropped. Can be filename or numpy array.
        label_data: N-dimensional array or str
            A label mask. Must be the same size as input_data
        mask_value: int or float
            Values equal to mask_value will be replaced with replacement_value.
        return_labels: float or list
            Label values to be preserved at the exclusion of other values.
        replacement_value: int or float
            Values equal to mask_value or not in return_labels will be replaced with replacement_value.

        Returns
        -------
        input_numpy: array
            Transformed input array.
    """

    input_numpy, label_numpy = convert_input_2_numpy(input_data), convert_input_2_numpy(label_data)

    input_numpy[label_numpy == mask_value] = replacement_value

    # if not isinstance(return_labels, list): 
    #     return_labels = [return_labels]

    # if len(return_labels) > 0:
    #     # [TODO FIX]
    #     input_numpy[label_numpy != return_labels[0]] = replacement_value

    if output_filename != '':
        if isinstance(input_data, basestring):
            save_numpy_2_nifti(input_numpy, input_data, output_filename)
        else:
            save_numpy_2_nifti(input_numpy, output_path=output_filename)
    else:
        return input_numpy

def run_test():
    return

if __name__ == '__main__':
    run_test()