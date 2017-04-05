""" A general module for normalization methods. Likely to
	be borrowing heavily from other packages. TODO: Fill in
	these methods.
"""

import numpy as np

def zero_mean_unit_variance(input_numpy=[], input_nifti=[], mask_numpy=[], mask_nifti=[], output_filename=[]):

    """ Normalization to zero mean and unit variance. Helpful preprocessing for deep learning
        applications. TODO: This function is our first function to take in both numpy and nifti
        files. Maybe make some sort of general function that serves as an interchange between the
        two in the utilities folder. Similar notes for output in nifti or numpy format.
    """

    if input_numpy == []:
        input_numpy = nifti_2_numpy(input_nifti)
    if mask_numpy == [] and mask_nifti != []:
        mask_numpy = nifti_2_numpy(mask_nifti)

    if mask_numpy == []:
        vol_mean = np.mean(normalize_numpy)
        vol_std = np.std(normalize_numpy)
        input_numpy = (input_numpy - vol_mean) / vol_std        
    else:
        masked_numpy = np.ma.masked_where(mask_numpy == 0, input_numpy)
        vol_mean = np.ma.mean(masked_numpy)
        vol_std = np.ma.std(masked_numpy)
        input_numpy = (masked_numpy - vol_mean) / vol_std
        input_numpy[mask_numpy == 0] = 0

    if output_filename != []:
        save_numpy_2_nifti(input_numpy, normalize_volume, output_filename)
    else:
        return input_numpy

if __name__ == '__main__':
	pass