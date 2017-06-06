import os
from shutil import copy

from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.threshold import crop_with_mask
from ..qtim_preprocessing.resample import resample
from ..qtim_preprocessing.normalization import zero_mean_unit_variance
from ..qtim_utilities.file_util import nifti_splitext, grab_files_recursive

def coregister_pipeline(study_name, base_directory, labelmap_volume, destination_volume='T2', config_file=None):

    """ This script is meant to coregister a series of volumes into the same space.

        One can either register all volumes to destination_volume, or register via a series of registrations
        specified in config_file.

        Parameters
        ----------
        study_name: str
            A QTIM study name code, usually three letters.
        base_directory: str
            The full path to the directory from which to search for studies. The study directory
            should be contained in this directory.
        skip_modalities: str or list of str
            Any modalities that should not be processed.

    """

    # TODO: Implement this with a config file.
    input_modality_dict = {'T2': ['T2SPACE.nii'],
                            'FLAIR': ['3D-FLAIR.nii', ]
                            'T1.nii', 'T1-Post.nii', 'FLAIR.nii', 'MPRAGE-Pre.nii', 'MPRAGE-Post.nii'],
                            'DCE': ['dce1_mc_ss.nii', 'dce1_mc.nii', 'dce1_mc_ss_mask.nii']
                            'DTI': ['FA.nii', 'L1.nii', 'L2.nii', 'L3.nii', 'MD.nii', 'M0.nii', 'S0.nii', 'sse.nii', 'V1.nii', 'V2.nii', 'V3.nii', 'diff_mc_ss.nii', 'diff_mc_ss_mask.nii']
                            'DSC': ['DSC_ge.nii', 'DSC_se.nii']}

    registration_tree = [['FLAIR', 'FLAIR-3D.nii'],
                            ['FLAIR-3D.nii', 'T2.nii'],
                            ['FLAIR', ]
                        ]

    coreg_volumes = []
    for folder in input_modality_dict:
        for modality in input_modality_dict[folder]:
            coreg_volumes += grab_files_recursive(os.path.join(base_directory, study_name, 'ANALYSIS', folder), '*' + modality + '*')

    if config_file is None:

        # Coregister to one volume.
        for coreg_volume in coreg_volumes:

            split_path = os.path.normpath(volume).split(os.sep)
            output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'DEEPLEARNING', split_path[-4], split_path[-3])

            # Don't register the destination_volume to itself..
            if destination_volume in coreg_volume:

                continue

            # Make sure a mask was created in the previous step.
            skull_strip_mask = os.path.join(output_folder, split_path[-4] + '-' + split_path[-3] + '-' + 'SKULL_STRIP_MASK.nii.gz')
            if not os.path.exists(skull_strip_mask):
                print 'No skull-stripping mask created, skipping this volume!'
                continue

            # Use existing mask to skull-strip if necessary.
            skull_strip_output = os.path.join(output_folder, nifti_splitext(dl_volume)[0] + '_ss' + nifti_splitext(dl_volume)[-1])
            if not os.path.exists(skull_strip_output):
                crop_with_mask(dl_volume, skull_strip_mask, output_filename=skull_strip_output)

            # Resample and remove previous file.
            resample_output = os.path.join(output_folder, nifti_splitext(skull_strip_output)[0] + '_iso' + nifti_splitext(skull_strip_output)[-1])
            if not os.path.exists(skull_strip_output):
                resample(skull_strip_output, resample_output, output_filename=skull_strip_output)
            os.remove(skull_strip_output)

            # Mean normalize and remove previous file.
            normalize_output = os.path.join(output_folder, nifti_splitext(dl_volume)[0] + '_DL' + nifti_splitext(dl_volume)[-1])
            if not os.path.exists(skull_strip_output):
                zero_mean_unit_variance(resample_output, normalize_output, output_filename=skull_strip_output)
            os.remove(resample_output)

    else:
        pass

    return

def run_test():
    pass

if __name__ == '__main__':
    run_test()