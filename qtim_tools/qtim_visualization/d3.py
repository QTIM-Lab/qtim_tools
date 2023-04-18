import os
import numpy as np

from shutil import copy
from bs4 import BeautifulSoup

from ..qtim_utilities.format_util import FORMAT_LIST, convert_input_2_numpy
from ..qtim_utilities.array_util import extract_maximal_slice, truncate_image
from ..qtim_utilities.image_util import save_numpy_2_img
from ..qtim_utilities.text_util import save_numpy_2_csv


def labeled_scatter(input_data, html_directory, image_col=0, labels=False, label_col=1, x_col=2, y_col=3, color_col=-1, data_columns=-1, dimensions=[], rescale=0, extract_maximal_slice_mode='', mask_value=0, save_labels=True, ignore_errors=True, overwrite=True):

    """ DOCUMENTATION TODO

        Also, this function should likely take a YAML config file.
    """

    # Check for csv input
    if isinstance(input_data, str):
        input_data = np.genfromtxt(input_data, delimiter=',', dtype=object)

    # Check for existence of destination folder.
    if not os.path.exists(html_directory):
        os.mkdir(html_directory)
    if not os.path.exists(os.path.join(html_directory, 'imgs')):
        os.mkdir(os.path.join(html_directory, 'imgs'))

    # Check for labels when picking the largest slice.
    if extract_maximal_slice_mode == '':
        if labels:
            extract_maximal_slice_mode = 'max_label'
        else:
            extract_maximal_slice_mode = 'max_intensity'

    # Create a new csv, required for modified image_paths.
    output_csv = np.copy(input_data).astype(object)

    for row_idx, row in enumerate(input_data):

        if row_idx == 0:
            continue

        filename = row[image_col]

        _, file_extension = os.path.splitext(filename)

        file_basename = os.path.basename(os.path.abspath(filename))

        if file_extension not in FORMAT_LIST['image']:

            output_filename = os.path.join(html_directory, 'imgs', str.split(file_basename, '.')[0] + '.png')

            if os.path.exists(output_filename) and not overwrite:
                continue

            array_3d = convert_input_2_numpy(filename)
            label_3d = []

            if labels:
                label_3d = convert_input_2_numpy(os.path.abspath(row[label_col]))

            maximal_slice, slice_index = extract_maximal_slice(array_3d, label_3d, mode=extract_maximal_slice_mode, return_index=True)

            maximal_slice[label_3d[:,:,slice_index] == mask_value] = 0

            maximal_slice = truncate_image(np.squeeze(maximal_slice))

            save_numpy_2_img(np.squeeze(maximal_slice), output_filename, dimensions, rescale)

        else:

            output_filename = os.path.join(html_directory, 'imgs', file_basename)

            copy(filename, output_filename)

        output_csv[row_idx, 0] = os.path.join('.','imgs', os.path.basename(output_filename))

    save_numpy_2_csv(output_csv, os.path.join(html_directory, 'viz_data.csv'))

    template_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'd3_templates', 'labeled_scatter.html')
    copy(template_file, os.path.join(html_directory, 'labeled_scatter.html'))

def run_test():
    return

if __name__ == '__main__':
    run_test()
