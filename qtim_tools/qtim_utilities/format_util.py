""" This will be a broader utility for interchanging between
    medical imaging (and perhaps other) formats.
"""

""" One might consider making a class for format converters for
    the future. Classes obfuscate the code a bit for new users,
    however, so for now we'll make it a TODO.
"""

# from nifti_util import nifti_2_numpy
from qtim_tools.qtim_utilities.dicom_util import dcm_2_numpy
from qtim_tools.qtim_utilities.nrrd_util import nrrd_2_numpy
from qtim_tools.qtim_utilities.image_util import img_2_numpy
from qtim_tools.qtim_utilities.nifti_util import nifti_2_numpy

import numpy as np
import nibabel as nib


def itk_transform_2_numpy(filepath):

    """ This function takes in an itk transform text file and converts into a 4x4
        array.

        TODO: Ensure this correctly rotates.
        TODO: Make work for more than just .txt files.

        Parameters
        ----------
        filepath: str
            The filepath to be converted

        Returns
        -------
        output_array: numpy array
            A 4x4 float matrix containing the affine transform.
    """

    with open(filepath) as f:
        content = f.readlines()

    for row_idx, row in enumerate(content):
        if row.startswith("Parameters:"):
            r_idx = row_idx
        if row.startswith("FixedParameters:"):
            t_idx = row_idx

    output_array = np.zeros((4,4))

    rotations = [float(r) for r in str.split(content[r_idx].replace("Parameters: ", '').rstrip(), ' ')]
    translations = [float(t) for t in str.split(content[t_idx].replace("FixedParameters: ", '').rstrip(), ' ')] + [1]

    for i in range(4):
        output_array[i,0:3] = rotations[i*3:(i+1)*3]
        output_array[i, 3] = translations[i]

    return output_array

# Consider merging these into one dictionary. Separating them
# is easier to visaulize though.
FORMAT_LIST = {'dicom':('.dcm','.ima'),
                'nifti':('.nii','.nii.gz'),
                'nrrd':('.nrrd','.nhdr'),
                'image':('.jpg','.png'),
                'itk_transform':('.txt')}

NUMPY_CONVERTER_LIST = {'dicom':dcm_2_numpy,
                'nifti':nifti_2_numpy,
                'nrrd': nrrd_2_numpy,
                'image':img_2_numpy}

def dicom_convert_slicer():

    return

def check_format(filepath):

    format_type = None

    for data_type in FORMAT_LIST:
        if filepath.lower().endswith(FORMAT_LIST[data_type]):
            format_type = data_type
        if format_type is not None:
            break

    if format_type is None:
        print('Error! Input file extension is not supported by qtim_tools. Returning None.')
    else:
        return format_type

def convert_input_2_numpy(input_data, input_format=None, return_header=False, return_type=False):
    
    """ Copies a file somewhere else. Effectively only used for compressing nifti files.

        Parameters
        ----------
        input_filepath: str
            Input filepath.
        return_header: bool
            If true, returns header information in nibabel format.

        Returns
        -------
        img: Numpy array
            Untransformed image data.
        header: list
            Varies from format to format.
        type: str
            Internal code for image type.
    """

    return_items = []

    if isinstance(input_data, str):
        if input_format is None:
            input_format = check_format(input_data)

        if input_format is None:
            print('Cannot understand input format for numpy conversion, returning None.')
            if return_header:
                return None, None
            else:
                return None

        if return_header:
            return_items += NUMPY_CONVERTER_LIST[input_format](input_data, return_header=True)
        else:
            return_items = [NUMPY_CONVERTER_LIST[input_format](input_data)]
        if return_type:
            return_items += [input_format]

    else:
        return_items += [input_data]
        if return_header:
            return_items += [None]
        if return_type:
            return_items += ['numpy']

    if len(return_items) > 1:
        return return_items
    else:
        return return_items[0]

def save_numpy_2_file(input, output_filename, reference_file=[], output_format=[]):
    return
