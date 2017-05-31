import subprocess
import os

from shutil import move

from ..qtim_utilities.nifti_util import save_numpy_2_nifti
from ..qtim_utilities.format_util import convert_input_2_numpy

def skull_strip(input_data, output_filename='', output_mask_filename='', method="bet", command="fsl4.1-bet2", temp_dir='./', extra_parameters={}):

    """ A catch-all function for skull-stripping. Will perform skull-stripping on an input volume
        depending on the 'method' and 'command' inputted. Will output a binary skull-mask to
        output_mask_filename if provided.

        Parameters
        ----------
        input_data: str or array
            Can be a 3D volume or a filename.
        output_filename: str
            Location to save output data to.
        output_mask_filename: str
            Location to save binary skull mask to.
        method: str
            Will perform motion correction according to the provided method.
            Currently available: ['fsl']
        command: str
            The literal command-line string to be inputted via Python's subprocess module.
        temp_dir: str
            If temporary files are created, they will be saved here.
        extra_parameters: dict
            A dictionary of method-specific parameters that one may submit.

        Returns
        -------
        output: array, array
            Output data, only if output_filename is left as ''.
    """

    skull_strip_methods = ['bet']
    if method not in skull_strip_methods:
        print 'Input \"method\" parameter is not available. Available methods: ', skull_strip_methods
        return

    if method == 'bet':

        # A good reason to have a Class for qtim methods is to cut through all of this extra code.

        temp_input, temp_output, temp_mask_output = False, False, False

        if not isinstance(input_data, basestring):
            input_filename = os.path.join(temp_dir, 'temp.nii.gz')
            save_numpy_2_nifti(input_data, input_filename)
            temp_input = True
        else:
            input_filename = input_data

        if output_filename == '':
            temp_output = True
            output_filename = os.path.join(temp_dir, 'temp_out.nii.gz')

        if output_mask_filename == '':
            temp_mask_output = True
            output_mask_filename = os.path.join(temp_dir, 'temp_mask_out.nii.gz')

        if extra_parameters['fsl_threshold'] is None:
            extra_parameters['fsl_threshold'] = .5

        print ' '.join([command, input_filename, output_filename, '-f', str(extra_parameters['fsl_threshold']), '-g', '0', '-m'])
        subprocess.call([command, input_filename, output_filename, '-f', str(extra_parameters['fsl_threshold']), '-g', '0', '-m'])

        if output_mask_filename != '':
            move(output_filename + '_mask.nii.gz', output_mask_filename)

        if temp_input:
            os.remove(input_filename)
            pass

        if temp_output or temp_mask_output:
            output, output_mask = convert_input_2_numpy(output_filename), convert_input_2_numpy(output_mask_filename)
            os.remove(output_filename)
            os.remove(output_mask_filename)
            return output, output_mask



def run_test():
    return

if __name__ == '__main__':
    run_test()