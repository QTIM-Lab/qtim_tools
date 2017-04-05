""" This will be a broader utility for interchanging between
    medical imaging (and perhaps other) formats.
"""



# This is magic code for Python 3 compatability. Of course
# this package isn't Python 3 compatible, but why not start now.
try:
  basestring
except NameError:
  basestring = str

FORMAT_LIST = {'dicom':['*.dcm','*.ima','*'],
                'nifti':['*.nii','*.nii.gz'],
                'nrrd':['*.nrrd','*.nhdr']}

def convert_input_2_numpy(input)
    if isinstance(input, basestring):
        pass
    else:
        return input

if __name__ == '__main__':
    pass