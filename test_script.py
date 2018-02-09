""" This is a continuously evolving test script for de-bugging purposes.
"""

import qtim_tools

import os
import numpy as np
import fnmatch
from shutil import copy

from qtim_tools.qtim_utilities.dicom_util import dcm_2_numpy, dcm_2_nifti

folder = 'C:/Users/azb22/Documents/Scripting/Prostate_Texture/Test'
outfile1 = 'C:/Users/azb22/Documents/Scripting/Prostate_Texture/no_normalize.csv'
outfile2 = 'C:/Users/azb22/Documents/Scripting/Prostate_Texture/normalize.csv'

qtim_tools.qtim_features.generate_feature_list_batch(folder, outfile=outfile1, labels=True, features=['GLCM','morphology', 'statistics'], levels = 100, mask_value = 0, erode = [0,0,0], overwrite = True, label_suffix='-label', set_label='', file_regex='*.nii*', recursive=False, normalize_intensities=False)
qtim_tools.qtim_features.generate_feature_list_batch(folder, outfile=outfile2, labels=True, features=['GLCM','morphology', 'statistics'], levels = 100, mask_value = 0, erode = [0,0,0], overwrite = True, label_suffix='-label', set_label='', file_regex='*.nii*', recursive=False, normalize_intensities=False)