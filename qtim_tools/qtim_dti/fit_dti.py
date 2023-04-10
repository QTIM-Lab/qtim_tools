import subprocess
import os

from ..qtim_utilities.nifti_util import save_numpy_2_nifti
from ..qtim_utilities.format_util import convert_input_2_numpy

def run_dtifit(input_data, input_bvec, input_bval, input_mask='', output_fileprefix='', method="fsl", command="fsl5.0-dtifit", temp_dir='./'):

    """ This will fail if used in numpy mode, currently.
    """

    motion_correction_methods = ['fsl']
    if method not in motion_correction_methods:
        print('Input \"method\" parameter is not available. Available methods: ', motion_correction_methods)
        return

    if method == 'fsl':

        # A good reason to have a Class for qtim methods is to cut through all of this extra code.

        temp_input, temp_output = False, False

        if not isinstance(input_data, basestring):
            input_filename = os.path.join(temp_dir, 'temp.nii.gz')
            save_numpy_2_nifti(input_data, input_filename)
            temp_input = True
        else:
            input_filename = input_data

        if output_fileprefix == '':
            temp_output = True
            output_fileprefix = os.path.join(temp_dir, 'temp_out.nii.gz')

        # TODO: Figure out what last parameter, reference number, means.
        print(' '.join([command, '-V', '--sse', '-k', input_filename, '-o', output_fileprefix, '-m', input_mask, '-r', input_bvec, '-b', input_bval]))
        subprocess.call([command, '-V', '--sse', '-k', input_filename, '-o', output_fileprefix, '-m', input_mask, '-r', input_bvec, '-b', input_bval])

        if temp_input:
            os.remove(input_filename)
            pass

        if temp_output:
            output = convert_input_2_numpy(output_fileprefix)
            os.remove(output_fileprefix)
            return output


def run_test():
    return

if __name__ == '__main__':
    run_test()
