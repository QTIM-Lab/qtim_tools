""" This module should be used for functions that null out values in an array
    based on a condition. Primarily used for masking.
"""

import numpy as np

from ..qtim_utilities.format_util import convert_input_2_numpy
from ..qtim_utilities.nifti_util import save_numpy_2_nifti

def resample(input_data, output_filename='', input_transform_file='', method="slicer", command="Slicer", temp_dir='./', param_dict={}):

    """ A catch-all function for resampling. Will resample a 3D volume to given dimensions according
        to the method provided.

        TODO: Add resampling for 4D volumes.
        TODO: Add dimension, interpolation, reference parameter. Currently set to linear/isotropic.

        Parameters
        ----------
        input_data: str or array
            Can be a 3D volume or a filename.
        output_filename: str
            Location to save output data to. If left as '', will return numpy array.
        input_transform_file: str
            detatails TBD, unimplemented
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

    skull_strip_methods = ['slicer']
    if method not in skull_strip_methods:
        print 'Input \"method\" parameter is not available. Available methods: ', skull_strip_methods
        return

    if method == 'slicer':

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

        ResampleVolume_base_command = ['Slicer', '--launch', 'ResampleScalarVolume', '-i', interpolation_mode]
        ResampleVolume_base_command += ['-s', str(dimensions).strip('[]').replace(' ', '')]
        ResampleVolume_specific_command = ResampleVolume_base_command + [resample_volume, output_filename]

        # TODO: Figure out what last parameter, reference number, means.
        print ' '.join([command, '--launch', 'ResampleScalarVolume', '-i', 'linear', 's', '1,1,1', input_filename, output_filename])
        subprocess.call([command, '--launch', 'ResampleScalarVolume', '-i', 'linear', 's', '1,1,1', input_filename, output_filename])

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