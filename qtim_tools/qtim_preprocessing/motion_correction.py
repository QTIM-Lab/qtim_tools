import subprocess
import os

from ..qtim_utilities.nifti_util import save_numpy_2_nifti
from ..qtim_utilities.format_util import convert_input_2_numpy

def motion_correction(input_data, output_filename='', method="fsl", command="fsl4.1-eddy_correct", temp_dir='./'):

    """
    """

    motion_correction_methods = ['fsl']
    if method not in motion_correction_methods:
        print 'Input \"method\" parameter is not available. Available methods: ', motion_correction_methods
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

        if output_filename == '':
            temp_output = True
            output_filename = os.path.join(temp_dir, 'temp_out.nii.gz')

        # TODO: Figure out what last parameter, reference number, means.
        print ' '.join([command, input_filename, output_filename, '0'])
        subprocess.call([command, input_filename, output_filename, '0'])

        if temp_input:
            os.remove(input_filename)
            pass

        if temp_output:
            output = convert_input_2_numpy(output_filename)
            os.remove(output_filename)
            return output


def run_test():
    return

if __name__ == '__main__':
    run_test()