""" This is a continuously evolving test script for de-bugging purposes.
"""

import qtim_tools

import os
import numpy as np
import fnmatch
from shutil import copy

from qtim_tools.qtim_utilities.dicom_util import dcm_2_numpy

dcm_2_numpy('/home/anderff/Documents/Data/TEST_DICOMS/ALL')