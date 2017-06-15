import os
from shutil import copy

from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.threshold import crop_with_mask
from ..qtim_preprocessing.resample import resample
from ..qtim_preprocessing.normalization import zero_mean_unit_variance
from ..qtim_preprocessing.registration import register_volume
from ..qtim_utilities.file_util import nifti_splitext, grab_files_recursive

def coregister_pipeline(study_name, base_directory, labelmap_volume, destination_volume='T2', overwrite=True, end_dimensions=None, resampled=True, config_file=None):

    """ This script is meant to coregister a series of volumes into the same space.

        One can either register all volumes to destination_volume, or register via a series of registrations
        specified in config_file.

        TODO: This is a classic tree coding challenge, would be fun to make it effecient.
        TODO: This is super ineffecient because there is a lot of input & output. In the future, chain transforms.

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
    input_modality_dict = {'T2': ['ANATOMICAL', ['T2SPACE.nii']]
                            'T1_Pre': ['ANATOMICAL', ['T1Pre.nii']],
                            'T1_Post': ['ANATOMICAL', ['T1Post.nii', 'T1Post-label.nii']],
                            'FLAIR': ['ANATOMICAL', ['FLAIR.nii', 'FLAIR-label.nii']],
                            '3D-FLAIR': ['ANATOMICAL', ['3D-FLAIR.nii']],
                            'MPRAGE_Pre': ['ANATOMICAL', ['MPRAGE-Pre.nii']],
                            'MPRAGE_Post': ['ANATOMICAL', ['MPRAGE-Post.nii']],
                            'DCE1': ['DCE', ['dce1_mc_ss.nii', 'dce1_mc.nii', 'dce1_mc_ss_mask.nii']],
                            'DCE2': ['DCE', ['dce2_mc_ss.nii', 'dce2_mc.nii', 'dce2_mc_ss_mask.nii']],
                            'DTI': ['DTI', ['diff_mc_ss.nii', 'FA.nii', 'L1.nii', 'L2.nii', 'L3.nii', 'MD.nii', 'M0.nii', 'S0.nii', 'sse.nii', 'V1.nii', 'V2.nii', 'V3.nii', 'diff_mc_ss_mask.nii']],
                            'DSC_GE': ['DSC', ['DSC_ge.nii']],
                            'DSC_SE': ['DSC', ['DSC_se.nii']]}

    # Order in which to register files.
    registration_tree = [['FLAIR', '3D-FLAIR', 'T2'],
                        ['MPRAGE_Pre', 'MPRAGE_Post', 'T2']
                        ]

    # Create Patient/VISIT Index based off of ANATOMICAL folder
    # Maybe make this a pre-built function.
    test_directory = os.path.join(base_directory, study_name, 'ANALYSIS', 'ANATOMICAL')
    patient_visit_list = []
    for patient_num in sorted(os.listdir(os.path.join(test_directory, '*/'))):
        for visit_num in sorted(os.listdir(os.path.join(test_directory, patient_num, '*/'))):
            patient_visit_list += [patient_num, visit_num]

    for patient_visit in patient_visit_list:

        # Get and create output folders.
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'COREGISTRATION', patient_visit[0], patient_visit[1])
        output_folder_not_resampled = os.path.join(base_directory, study_name, 'ANALYSIS', 'COREGISTRATION', patient_visit[0], patient_visit[1], 'NOT_RESAMPLED')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(output_folder_not_resampled):
            os.makedirs(output_folder_not_resampled)

        # Track transforms...
        transform_dictionary = {}

        # Iterate through registration tests
        for registration_pathway in registration_tree:

            for reg_idx, moving_step in enumerate(registration_pathway[0:-1]):

                # Get the fixed volume label
                fixed_step = registration_pathway[reg_idx+1]

                # Find the fixed volume
                fixed_volume = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[fixed_step][0], patient_visit[0], patient_visit[1]), '*' + input_modality_dict[fixed_step][1][0] + '*')
                if fixed_volume is None:
                    print 'Missing ', moving_step, 'in registration pathway', registration_pathway, '. Skipping this pathway.'
                    continue

                # Get available files to register. Reformat so this is less redundant.
                moving_volumes = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[moving_step][0], patient_visit[0], patient_visit[1]), '*')

                # Get the volume that is actually going to be registered.
                leader_moving_volume = [volume for volume in moving_volumes if input_modality_dict[moving_step][1][0] in volume]
                if moving_volume is None:
                    print 'Missing ', fixed_step, 'in registration pathway', registration_pathway, '. Skipping this pathway.'
                    continue        

                # If you're at the last step, register for real.
                if reg_idx == len(registration_pathway - 2):
                    register_volumes(leader_moving_volume, fixed_volume, )

                # Find the moving volume(s). The following steps are more elaborate than needed.
                all_moving_volumes = []

                # Single volume case.          
                if registration_mode == 'single':
                    moving_volumes = [registration_volume for registration_volume in registration_files if moving_step in registration_volume]
                    if len(moving_volumes) > 1:
                        print 'Multiple moving volumes found for step', registration_pathway[reg_idx:reg_idx+2], '- aborting this pathway.'
                        continue

                # Multiple volume case.
                elif registration_mode == 'multiple':
                    multiple_moving_steps = input_modality_dict[moving_step]
                    for single_moving_step in multiple_moving_steps:
                        moving_volumes = [registration_volume for registration_volume in registration_files if single_moving_step in registration_volume]

                # Error-checking, no volumes found.
                if len(moving_volumes) == 0:
                    print 'Moving volume not found for step', registration_pathway[reg_idx:reg_idx+2], '- aborting this pathway.'
                    continue

                # Get output filenames.
                moving_suffix, target_suffix = get_file_suffixes(moving_volume[0], target_volume)
                output_transform = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'_o.nii.gz')
                output_volume = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.txt')
                output_volume_resampled = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.nii.gz')

                # Register first volume.
                if not os.path.exists(output_transform):
                    register_volume(moving_volume[0], target_volume, output_filename=output_volume_resampled, output_transform=output_transform)
                else:
                    # get right command
                    apply_affine_transform(moving_volume[0], output_transform, output_filename=output_volume_resampled)

                # If applicable, move over the rest of the volumes.
                if len(moving_volume) > 1:
                    for additional_volume in moving_volumes[1:]:
                        moving_suffix, target_suffix = get_file_suffixes(additional_volume, target_volume)
                        output_volume = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.txt')
                        output_volume_resampled = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + target_suffix +'.nii.gz')
                        apply_affine_transform(moving_volume[0], output_transform, output_filename=output_volume_resampled)


    # Useful list to have for future steps.
    all_modality_list = []
    for key in input_modality_dict:
        all_modality_list += input_modality_dict[key]

    # Folders to search in base_directory/ANALYSIS
    modality_folder = ['ANATOMICAL', 'DSC', 'DTI', 'DCE', 'SUV']

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