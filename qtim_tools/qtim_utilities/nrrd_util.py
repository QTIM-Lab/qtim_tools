""" Utilities for dealing with NRRD files. So far, this is mostly
	a nifti library, so these will probably just be conversion
	utilities. The existence of this library should imply the creation
	of an array_util, as many of the functions in nifti_util are not
	specific to niftis.
"""

import nibabel as nib
import numpy as np
import nrrd

def nrrd_2_numpy(input_nrrd, return_header=False):
	
	""" Loads nrrd data and optionally return a nrrd header
		in pynrrd's format. If array is 4D, swaps axes so
		that time dimension is last to match nifti standard.
	"""

	nrrd_data, nrrd_options = nrrd.read(input_nrrd)

	if nrrd_data.ndim == 4:
		nrrd_data = np.rollaxis(nrrd_data, 0, 4)

	if return_header:
		return nrrd_data, nrrd_options
	else:
		return nrrd_data

def save_numpy_2_nrrd(input_numpy, reference_nrrd=[], output_filepath=''):
    return