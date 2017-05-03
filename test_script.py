from qtim_tools import qtim_features

# qtim_tools.qtim_dce.tofts_parametric_mapper.test_method_2d()

import os
import numpy as np
import fnmatch
from shutil import copy

intensity_squares_filepath = qtim_features.phantoms.get_phantom_filepath('intensity_square')

qtim_features.generate_feature_list_batch(intensity_squares_filepath, outfile='intensity_square_phantom.csv', labels=True)
