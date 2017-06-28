""" This is a continuously evolving test script for de-bugging purposes.
"""

import qtim_tools

import os
import numpy as np
import fnmatch
from shutil import copy

# glcm_filepath = qtim_tools.qtim_features.phantoms.get_phantom_filepath('glcm_square')

# qtim_tools.qtim_features.generate_feature_list_batch(glcm_filepath, outfile='visualization_test.csv', labels=True)

# qtim_tools.qtim_visualization.d3_models.labeled_scatter(input_data='visualization_test.csv', html_directory='/home/anderff/Documents/MGH/Visualizations/qtim_tools_test_dir', labels=True, dimensions=[200,200])

# print qtim_tools.qtim_utilities.file_util.grab_linked_file('./setup.py', suffix=".py", return_multiple=True, recursive=True)

qtim_tools.qtim_dce.dce_util.create_gradient_phantom('test')