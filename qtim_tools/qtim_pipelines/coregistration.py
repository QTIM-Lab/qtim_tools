import os
from shutil import copy

from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.threshold import crop_with_mask
from ..qtim_preprocessing.resample import resample
from ..qtim_preprocessing.normalization import zero_mean_unit_variance
from ..qtim_preprocessing.registration import register_volumes
from ..qtim_utilities.file_util import nifti_splitext, grab_files_recursive

def coregister_pipeline(study_name, base_directory, labelmap_volume, destination_volume='T2', end_dimensions=[], resampled=True, config_file=None):

    """ This script is meant to coregister a series of volumes into the same space.

        One can either register all volumes to destination_volume, or register via a series of registrations
        specified in config_file.

        TODO: This is a classic tree coding challenge, would be fun to make it effecient.

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
                            'T1Pre': [],
                            'T1Post': [],
                            'FLAIR': ['FLAIR.nii'],
                            '3D-FLAIR': ['3D-FLAIR.nii'],
                            'MPRAGE_Pre': ['MPRAGE-Pre.nii'],
                            'MPRAGE_Post': ['MPRAGE-Post.nii'],
                            'DCE1': ['dce1_mc_ss.nii', 'dce1_mc.nii', 'dce1_mc_ss_mask.nii'],
                            'DCE2': ['dce2_mc_ss.nii', 'dce2_mc.nii', 'dce2_mc_ss_mask.nii'],
                            'DTI': ['diff_mc_ss.nii', 'FA.nii', 'L1.nii', 'L2.nii', 'L3.nii', 'MD.nii', 'M0.nii', 'S0.nii', 'sse.nii', 'V1.nii', 'V2.nii', 'V3.nii', 'diff_mc_ss_mask.nii'],
                            'DSC_GE': ['DSC_ge.nii'],
                            'DSC_SE': ['DSC_se.nii']}

    # Useful list to have for future steps.
    all_modality_list = []
    for key in input_modality_dict:
        all_modality_list += input_modality_dict[key]

    # Folders to search in base_directory/ANALYSIS
    modality_folder = ['ANATOMICAL', 'DSC', 'DTI', 'DCE', 'SUV']

    # Order in which to register files.
    registration_tree = [['FLAIR', 'FLAIR-3D.nii', 'T2.nii'],
                            ['FLAIR']
                        ]

    # Grab all volumes.
    for folder in modality_folders:
        for modality in input_modality_dict[folder]:
            folder_niftis = grab_files_recursive(os.path.join(base_directory, study_name, 'ANALYSIS', folder), '*.nii*')

    # Grab a list of available patient data from the anatomical folder.
    # TODO: make a more robust method to do these calculations.
    patient_visit_data = {}
    for folder in glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', 'ANATOMICAL', '*/')):
        patient_num = os.path.basename(folder)
        for subfolder in glob.glob(os.path.join(folder, '*/')):
            visit_num = os.path.basename(subfolder)
            patient_visit_data[patient_num + '-' + visit_num] = glob.glob(os.path.join(subfolder, '*.nii*'))

    for patient_visit in patient_visit_data:

        # Get and create output folder.
        split_path = os.path.normpath(patient_visit).split('_')
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'COREGISTRATION', split_path[0], split_path[1])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Track transforms...
        transform_dictionary = {}

        # Iterate through registration tests
        for moving_step, target_step in registration_tree:

            # Get available files to register
            registration_files = patient_visit_data[patient_visit]

            # Check if one-to-one or all-to-one registration
            if '.nii' not in moving_step:
                registration_mode = 'multiple'
            else:
                registration_mode = 'single'

            # Error check in case of one-to-all registration.
            if '.nii' in target_step:
                print 'Registration step', registration_step, 'has problems. Cannot register one volume to a folder.'
                continue

            # Get target and moving volume from available files.
            # TODO: Add a bit of error-checking.
            target_volume = [registration_volume for registration_volume in registration_files if target_step in registration_volume]
            all_moving_volumes = []            
            if registration_mode == 'single':
                moving_volume = [registration_volume for registration_volume in registration_files if moving_step in registration_volume][0]
            elif registration_mode == 'multiple':
                multiple_moving_steps = input_modality_dict[moving_step]
                for single_moving_step in multiple_moving_steps:
                    all_moving_volumes = [registration_volume for registration_volume in registration_files if single_moving_step in registration_volume]
                    moving_volume = all_moving_volumes[0]

            # Get output suffixes
            moving_suffix, target_suffix = get_file_suffixes(moving_volume, target_volume)

            # Get output filenames
            output_transform = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'_o.nii.gz')
            output_volume = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.txt')
            output_volume_resampled = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.nii.gz')

            # Do the registration!
            # TODO: Conditionally not output volume.
            register_volumes(target_volume, moving_volume, output_volume, output_volume_resampled, output_transform)






    if config_file is None:

        # Coregister to one volume.
        for coreg_volume in coreg_volumes:

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

def get_file_suffixes(moving_volume, target_volume):

    moving_suffix = str.split(os.path.basename(moving_volume), patient_visit)[-1]
    moving_suffix = str.split(moving_suffix, '.')[0]
    target_suffix = str.split(os.path.basename(target_volume), patient_visit)[-1]
    target_suffix = str.split(target_suffix, '.')[0]
    
    return moving_suffix, target_suffix

def run_test():
    pass

if __name__ == '__main__':
    run_test()