""" This will be a broader utility for interchanging between
    medical imaging (and perhaps other) formats.
"""

""" One might consider making a class for format converters for
    the future. Classes obfuscate the code a bit for new users,
    however, so for now we'll maek it a TODO.
"""
from nifti_util import nifti_2_numpy
from dicom_util import dcm_2_numpy
from nrrd_util import nrrd_2_numpy
from image_util import img_2_numpy

# This is magic code for Python 3 compatability. Of course
# this package isn't Python 3 compatible, but why not start now.
try:
  basestring
except NameError:
  basestring = str

# Consider merging these into one dictionary. Separating them
# is easier to visaulize though.
FORMAT_LIST = {'dicom':('.dcm','.ima'),
                'nifti':('.nii','.nii.gz'),
                'nrrd':('.nrrd','.nhdr'),
                'image':('.jpg','.png')}

NUMPY_CONVERTER_LIST = {'dicom':dcm_2_numpy,
                'nifti':nifti_2_numpy,
                'nrrd': nrrd_2_numpy,
                'image':img_2_numpy}

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

def convert_input_2_numpy(input_data, input_format=[]):
    
    """ This function is meant to take in any normal imaging file 
        format and convert it into numpy. Numpy is supposed from
        now on to be the lingua franca of qtim_tools.
    """

    if isinstance(input_data, basestring):
        if input_format == []:
            input_format = check_format(input_data)

        if input_format == []:
            return []

        return NUMPY_CONVERTER_LIST[input_format](input_data)

    else:
        return input_data

def save_numpy_2_file(input, output_filename, reference_file=[], output_format=[]):
    return
