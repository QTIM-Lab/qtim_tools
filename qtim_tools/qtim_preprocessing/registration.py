""" This is a wrapper script for BRAINSFit registration by 3D Slicer. In the future, there could be an all-Python
    implementation of registration in this function. In the meantime, one will need 3D Slicer (or a Docker container
    with 3DSlicer inside).
"""

import numpy as np
import glob
import os
from subprocess import call
from shutil import copy

def register_all_to_one(fixed_volume, moving_volume_folder, output_folder='', output_suffix = '_r', file_regex='*.nii*', exclusion_phrase='label', software='slicer_brainsfit',Slicer_Path='Slicer', transform_type='Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine', transform_mode = 'useMomentsAlign', interpolation_mode = 'Linear', sampling_percentage = .02):

    """ Registers a folder of volumes to one pre-specified volumes using the BRAINSFit module in 3DSlicer. Must already have 3DSlicer installed.
    """

    if output_folder == '':
        output_folder = moving_volume_folder

    BRAINSFit_base_command = [Slicer_Path,'--launch', 'BRAINSFit', '--fixedVolume', '"' + fixed_volume + '"', '--transformType', transform_type, '--initializeTransformMode', transform_mode, '--interpolationMode', interpolation_mode, '--samplingPercentage', str(sampling_percentage)]

    moving_volumes = glob.glob(os.path.join(moving_volume_folder + file_regex))

    if len(moving_volumes) == 0:
        print 'No nifti volumes found in provided moving volume folder. Skipping this folder: ' + moving_volume_folder

    if exclusion_phrase != '':
        moving_volumes = [x for x in moving_volumes if exclusion_phrase not in x]

    for moving_volume in moving_volumes:

        no_path = str.split(moving_volume ,'\\')
        file_prefix = str.split(no_path[-1], '.nii')[0]

        if '.nii.gz' in moving_volume:
            output_filename = output_folder + file_prefix + output_suffix + '.nii.gz'
        else:
            output_filename = output_folder + file_prefix + output_suffix + '.nii'

        BRAINSFit_specific_command = BRAINSFit_base_command + ['--movingVolume','"' + no_path[0] +  '/' + no_path[1] + '"','--outputVolume','"' + output_filename + '"']

        try:
            print ' '.join(BRAINSFit_specific_command)
            call(' '.join(BRAINSFit_specific_command), shell=True)
        except:
            print 'Registration command failed. Did you provide the correct path to your Slicer insallation? Provided Slicer Path: ' + Slicer_Path
            pass

    return

def register_volume(moving_volume, fixed_volume, output_volume_filename=None, output_transform_filename=None, method='slicer_brainsfit', Slicer_Path='Slicer', transform_type='Rigid,ScaleVersor3D,ScaleSkewVersor3D,Affine', transform_mode = 'useMomentsAlign', interpolation_mode = 'Linear', sampling_percentage = .02):

    """ Registers a folder of volumes to one pre-specified volumes using the BRAINSFit module in 3DSlicer. Must already have 3DSlicer installed.
    """

    # A good reason to have a Class for qtim methods is to cut through all of this extra code.


    if method == 'slicer_brainsfit':

        BRAINSFit_command = [Slicer_Path,'--launch', 'BRAINSFit', '--fixedVolume', '"' + fixed_volume + '"', '--movingVolume', '"' + moving_volume + '"', '--transformType', transform_type, '--initializeTransformMode', transform_mode, '--interpolationMode', interpolation_mode, '--samplingPercentage', str(sampling_percentage)]

        if output_volume_filename is not None:
            BRAINSFit_command += ['--outputVolume', output_volume_filename]

        if output_transform_filename is not None:
            BRAINSFIT_command += ['--outputTransform', output_transform_filename]

        print ' '.join(BRAINSFit_command)
        subprocess.call([BRAINSFit_command])

        return

if __name__ == "__main__":
    pass