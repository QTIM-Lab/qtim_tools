""" This will be a broader utility for interchanging between
    medical imaging (and perhaps other) formats.
"""

""" One might consider making a class for format converters for
    the future. Classes obfuscate the code a bit for new users,
    however, so for now we'll make it a TODO.
"""

# from nifti_util import nifti_2_numpy
from dicom_util import dcm_2_numpy
from nrrd_util import nrrd_2_numpy
from image_util import img_2_numpy
from nifti_util import nifti_2_numpy

import numpy as np
import nibabel as nib

# This is magic code for Python 3 compatability. Of course
# this package isn't Python 3 compatible, but why not start now.
try:
  basestring
except NameError:
  basestring = str

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
    format_type = []

    for data_type in FORMAT_LIST:
        if filepath.lower().endswith(FORMAT_LIST[data_type]):
            format_type = data_type
        if format_type != []:
            break

    if format_type == []:
        print 'Error! Input file extension is not supported by qtim_tools. Returning [].'
    else:
        return format_type

def convert_input_2_numpy(input_data, input_format=[], return_header=False):
    
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
            A two item list. The first is the affine matrix in array format, the
            second is 

    """

    if isinstance(input_data, basestring):
        if input_format == []:
            input_format = check_format(input_data)

        if input_format == []:
            print 'Cannot understand input format for numpy conversion, returning None.'
            if return_header:
                return None, None
            else:
                return None

        if return_header:
            return NUMPY_CONVERTER_LIST[input_format](input_data, return_header=True)
        else:
            return NUMPY_CONVERTER_LIST[input_format](input_data)

    else:
        if return_header:
            return input_data, None
        else:
            return input_data

def save_numpy_2_file(input, output_filename, reference_file=[], output_format=[]):
    return
