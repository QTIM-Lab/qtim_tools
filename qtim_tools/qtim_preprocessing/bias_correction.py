import subprocess
import os

from ..qtim_utilities.nifti_util import save_numpy_2_nifti
from ..qtim_utilities.format_util import convert_input_2_numpy

def motion_correction(input_data, output_filename='', method="ants", command="N4BiasFieldCorrection", temp_dir='./'):

    """ A catch-all function for motion correction. Will perform motion correction on an input volume
        depending on the 'method' and 'command' inputted.

        Parameters
        ----------
        input_data: str or array
            Can be a 4D volume or a filename.
        output_filename: str
            Location to save output data to. If left as '', will return numpy array.
        method: str
            Will perform motion correction according to the provided method.
            Currently available: ['fsl']
        command: str
            The literal command-line string to be inputted via Python's subprocess module.
        temp_dir: str
            If temporary files are created, they will be saved here.

        Returns
        -------
        output: array
            Output data, only if output_filename is left as ''.
    """

    bias_correction_methods = ['ants', 'slicer']
    if method not in bias_correction_methods:
        print 'Input \"method\" parameter is not available. Available methods: ', bias_correction_methods
        return

    if method == 'ants':

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
        print ' '.join([command, '-i', input_filename, '-o', output_filename])
        subprocess.call([command, '-i', input_filename, '-o', output_filename])

        if temp_input:
            os.remove(input_filename)
            pass

        if temp_output:
            output = convert_input_2_numpy(output_filename)
            os.remove(output_filename)
            return output

    if method == 'slicer':

        print 'Slicer method not yet implemented! Sorry...'


def run_test():
    return

if __name__ == '__main__':
    run_test()