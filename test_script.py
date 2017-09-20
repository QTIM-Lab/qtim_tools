""" This is a continuously evolving test script for de-bugging purposes.
"""

import qtim_tools

from qtim_tools.qtim_utilities.dicom_util import dcm_2_numpy, dcm_2_nifti

dcm_2_nifti('INPUT_FOLDER', 'OUTPUT_FOLDER')