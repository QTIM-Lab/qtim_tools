""" This module should be used for functions that null out values in an array
    based on a condition. Primarily used for masking.
"""

import numpy as np

def crop_with_mask(input_numpy, mask_numpy, mask_value=0, replacement_value=0):

    input_numpy[mask_numpy == mask_value] = replacement_value

    return input_numpy

def run_test():
    return

if __name__ == '__main__':
    run_test()