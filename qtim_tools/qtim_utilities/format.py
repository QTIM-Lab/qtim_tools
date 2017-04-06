""" This will be a broader utility for interchanging between
    medical imaging (and perhaps other) formats.
"""

from nifti_util import nifti_2_numpy
from dicom_util import dcm_2_numpy

# This is magic code for Python 3 compatability. Of course
# this package isn't Python 3 compatible, but why not start now.
try:
  basestring
except NameError:
  basestring = str

FORMAT_LIST = {'dicom':['.dcm','.ima'],
                'nifti':['.nii','.nii.gz'],
                'nrrd':['.nrrd','.nhdr'],
                'image':['.jpg','.png']}

def check_format(filepath):

    return



def convert_input_2_numpy(input, input_format=[]):
    
    """ This function is meant to take in any normal imaging file 
        format and convert it into numpy. Numpy is supposed from
        now on to be the lingua franca of qtim_tools.
    """

    if isinstance(input, basestring):
        if input_format == []:
            format = check_format(input)
        pass
    else:
        return input

if __name__ == '__main__':
    pass