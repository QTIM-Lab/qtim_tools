""" This is a continuously evolving test script for de-bugging purposes.
"""

import qtim_tools

import os
import numpy as np
import fnmatch
from shutil import copy

from qtim_tools.qtim_utilities.dicom_util import dcm_2_numpy, dcm_2_nifti

dcm_2_nifti('C:/Users/abeers/Documents/Data/ME', 'C:/Users/abeers/Documents/Data/ME/Niftis')