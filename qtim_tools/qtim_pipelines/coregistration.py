import os
import glob
import numpy as np
from shutil import copy

from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.threshold import crop_with_mask
from ..qtim_preprocessing.resample import resample
from ..qtim_preprocessing.normalization import zero_mean_unit_variance
from ..qtim_preprocessing.registration import register_volume

from ..qtim_utilities.nifti_util import get_nifti_affine, set_nifti_affine, save_numpy_2_nifti
from ..qtim_utilities.file_util import nifti_splitext, grab_files_recursive, replace_suffix
from ..qtim_utilities.format_util import itk_transform_2_numpy
from ..qtim_utilities.transform_util import save_affine, generate_identity_affine, compose_affines, itk_2_vtk_transform

def coregister_pipeline(study_name, base_directory, destination_volume='T2', output_analysis_dir="TEST_COREGISTRATION", overwrite=True, end_dimensions=None, resampled=True, not_resampled=True, transforms=True, config_file=None, error_file=None):

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

    ### PBR #####
    # input_modality_dict = {'T2': ['ANATOMICAL', ['T2SPACE.nii']],
    #                         'T1_Pre': ['ANATOMICAL', ['T1Pre.nii']],
    #                         'T1_Post': ['ANATOMICAL', ['T1Post.nii', 'T1Post-label.nii']],
    #                         'FLAIR': ['ANATOMICAL', ['FLAIR.nii', 'FLAIR-label.nii']],
    #                         '3D-FLAIR': ['ANATOMICAL', ['3D-FLAIR.nii']],
    #                         'MEMPRAGE_Pre': ['ANATOMICAL', ['MEMPRAGE_Pre.nii']],
    #                         'MEMPRAGE_Post': ['ANATOMICAL', ['MEMPRAGE_POST.nii']],
    #                         'DCE1': ['DCE', ['dce1_mc.nii', 'dce1_mc_ss.nii', 'dce1_mc_ss_mask.nii']],
    #                         'DCE2': ['DCE', ['dce2_mc.nii', 'dce2_mc_ss.nii', 'dce2_mc_ss_mask.nii']],
    #                         'DTI': ['DTI', ['diff_mc.nii', 'diff_mc_ss.nii', 'FA.nii', 'L1.nii', 'L2.nii', 'L3.nii', 'MD.nii', 'M0.nii', 'S0.nii', 'sse.nii', 'V1.nii', 'V2.nii', 'V3.nii', 'diff_mc_ss_mask.nii']],
    #                         'DSC_GE': ['DSC', ['DSC_ge.nii']],
    #                         'DSC_SE': ['DSC', ['DSC_se.nii']]}

    # # Order in which to register files.
    # registration_tree = [['FLAIR', 'T2'],
    #                     ['DSC_GE', 'T2'],
    #                     ['DSC_SE', 'T2'],
    #                     ['DCE2', 'DCE1', 'T1_Pre', 'T1_Post', 'MEMPRAGE_Pre', 'MEMPRAGE_Post', 'T2']
    #                     ]

    # label_volumes = ['FLAIR-label.nii', 'T1Post-label.nii']

    # difficult_registration_files = ['DCE1', 'DCE2', 'DSC_GE', 'DSC_SE', 'DTI']

    # time_volumes = ['DCE1', 'DCE2', 'DSC_GE', 'DSC_SE', 'DTI']

    # name_change_dict = []

    # patient_directory = 'ANATOMICAL'

    #### NHX ####
    input_modality_dict = {'T2': ['RAW', ['T2SPACE.nii']],
                            # 'T1_Pre': ['ANATOMICAL', ['T1Pre.nii']],
                            'T1_Post': ['RAW', ['T1Post.nii', 'T1Post-label.nii']],
                            'FLAIR': ['RAW', ['FLAIR.nii', 'FLAIR-label.nii']],
                            # '3D-FLAIR': ['ANATOMICAL', ['3D-FLAIR.nii']],
                            # 'MEMPRAGE_Pre': ['ANATOMICAL', ['MEMPRAGE_Pre.nii']],
                            'MEMPRAGE_Post': ['RAW', ['MEMPRAGE_POST.nii']],
                            # 'DCE1': ['DCE', ['dce1_mc.nii', 'dce1_mc_ss.nii', 'dce1_mc_ss_mask.nii']],
                            # 'DCE2': ['DCE', ['dce2_mc.nii', 'dce2_mc_ss.nii', 'dce2_mc_ss_mask.nii']],
                            # 'DTI': ['DTI', ['diff_mc.nii', 'diff_mc_ss.nii', 'FA.nii', 'L1.nii', 'L2.nii', 'L3.nii', 'MD.nii', 'M0.nii', 'S0.nii', 'sse.nii', 'V1.nii', 'V2.nii', 'V3.nii', 'diff_mc_ss_mask.nii']],
                            'DSC_GE': ['RAW', ['DSC_ge.nii']],
                            'DSC_SE': ['RAW', ['DSC_se.nii']]}

    # Order in which to register files.
    registration_tree = [['FLAIR', 'T2'],
                        ['DSC_GE', 'T2'],
                        ['DSC_SE', 'T2'],
                        ['T1_Post', 'MEMPRAGE_Post', 'T2']
                        ]

    label_volumes = ['FLAIR-label.nii', 'T1Post-label.nii']

    difficult_registration_files = ['DCE1', 'DCE2', 'DSC_GE', 'DSC_SE', 'DTI']

    time_volumes = ['DCE1', 'DCE2', 'DSC_GE', 'DSC_SE', 'DTI']

    name_change_dict = {'RAW': {'dsc_ge.nii': 'DSC_ge.nii',
                                'dsc_se.nii': 'DSC_se.nii',
                                't1axialpostroi.nii': 'T1Post-label.nii',
                                't1axialpost.nii': 'T1Post.nii',
                                't2space.nii': 'T2SPACE.nii',
                                'memprage.nii': 'MEMPRAGE_Post.nii',
                                'flair.nii': 'FLAIR.nii',
                                'flairroi.nii': 'FLAIR-label.nii'
                                }
                                }

    patient_directory = 'RAW'

    # Create Patient/VISIT Index based off of ANATOMICAL folder
    # Maybe make this a pre-built function.
    test_directory = os.path.join(base_directory, study_name, 'ANALYSIS', patient_directory)
    patient_visit_list = []
    for patient_num in sorted(os.listdir(test_directory)):
        for visit_num in sorted(os.listdir(os.path.join(test_directory, patient_num))):
            patient_visit_list += [[patient_num, visit_num]]

    for patient_visit in patient_visit_list:

        # Get and create output folders.
        visit_code = '-'.join(patient_visit) + '-'
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', output_analysis_dir, patient_visit[0], patient_visit[1])
        output_folder_not_resampled = os.path.join(base_directory, study_name, 'ANALYSIS', output_analysis_dir, patient_visit[0], patient_visit[1], 'NOT_RESAMPLED')
        output_folder_transform = os.path.join(base_directory, study_name, 'ANALYSIS', output_analysis_dir, patient_visit[0], patient_visit[1], 'TRANSFORMS')

        for dir_check in [output_folder, output_folder_not_resampled, output_folder_transform]:
            if not os.path.exists(dir_check):
                os.makedirs(dir_check)

        print 'patient_visit', patient_visit

        file_deletion_list = []
        for name_change_directory in name_change_dict:
            for name_change_file in name_change_dict[name_change_directory]:

                print name_change_file
                print os.path.join(base_directory, study_name, 'ANALYSIS', name_change_directory, patient_visit[0], patient_visit[1], '*' + name_change_file + '*')

                name_change_volume = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', name_change_directory, patient_visit[0], patient_visit[1], '*' + name_change_file + '*'))

                # Error check the fixed volume
                if name_change_volume == []:
                    continue
                name_change_volume = name_change_volume[0]

                print name_change_volume

                if not os.path.basename(name_change_volume).startswith(visit_code):
                    copy_path = os.path.join(base_directory, study_name, 'ANALYSIS', name_change_directory, patient_visit[0], patient_visit[1], visit_code + name_change_dict[name_change_directory][name_change_file])
                else:
                    copy_path = os.path.join(base_directory, study_name, 'ANALYSIS', name_change_directory, patient_visit[0], patient_visit[1], name_change_dict[name_change_directory][name_change_file])
                
                if not os.path.exists(copy_path):
                    copy(name_change_volume, copy_path)

                print visit_code
                print copy_path

                file_deletion_list += [copy_path]

        # Iterate through registration tests
        for registration_pathway in registration_tree:

            # This first loop gets all the transforms and saves them in the transform dictionary.
            transform_dictionary = {}

            for reg_idx, moving_step in enumerate(registration_pathway[0:-1]):

                transform_dictionary[moving_step] = []

                # Get the fixed volume label
                fixed_step = registration_pathway[reg_idx+1]

                print '\n'

                print 'fixed_step', fixed_step

                print os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[fixed_step][0], patient_visit[0], patient_visit[1], '*' + input_modality_dict[fixed_step][1][0] + '*')

                # Find the fixed volume
                fixed_volume = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[fixed_step][0], patient_visit[0], patient_visit[1], '*' + input_modality_dict[fixed_step][1][0] + '*'))
                # Error check the fixed volume
                if fixed_volume == []:
                    print 'Missing', input_modality_dict[fixed_step][1][0], 'in registration pathway', registration_pathway, '. Skipping this step.'
                    continue
                fixed_volume = fixed_volume[0]

                print 'fixed_volume', fixed_volume

                # Get available files to register. Reformat so this is less redundant.
                moving_volume = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[moving_step][0], patient_visit[0], patient_visit[1], '*' + input_modality_dict[moving_step][1][0] + '*'))
                if moving_volume == []:
                    print 'Missing', input_modality_dict[moving_step][1][0], 'in registration pathway', registration_pathway, '. Skipping this step.'
                    continue        
                moving_volume = moving_volume[0]

                print 'leader_moving_volume', moving_volume

                if moving_step in difficult_registration_files:
                    sampling_percentage = 0.2
                else:
                    sampling_percentage = 0.02

                # # Get output filenames.
                moving_suffix, fixed_suffix = get_file_suffixes(moving_volume, fixed_volume, visit_code)
                output_transform = os.path.join(output_folder_transform, visit_code + moving_suffix + '_r_' + fixed_suffix +'.txt')

                # Create transforms
                if not os.path.exists(output_transform):
                    register_volume(moving_volume, fixed_volume, output_transform_filename=output_transform, Slicer_Path='/opt/Slicer-4.5.0-1-linux-amd64/Slicer', sampling_percentage=sampling_percentage)

                transform_dictionary[moving_step] = output_transform

            # Now do the actual transformations.
            for reg_idx, moving_step in enumerate(registration_pathway[0:-1]):

                np.set_printoptions(suppress=True)

                transform_list = [transform_dictionary[transform_step] for transform_step in registration_pathway[reg_idx:-1] if transform_dictionary[transform_step] != []]
                print transform_list
                if transform_list == []:
                    continue

                final_transform = generate_identity_affine()
                for concat_transform in transform_list:
                    print itk_transform_2_numpy(concat_transform)
                    print itk_2_vtk_transform(itk_transform_2_numpy(concat_transform))                    
                    final_transform = compose_affines(final_transform, itk_2_vtk_transform(itk_transform_2_numpy(concat_transform)))

                combined_transforms = []

                print 'transform_list', transform_list

                # Find the fixed volume
                reference_volume = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[registration_pathway[-1]][0], patient_visit[0], patient_visit[1], '*' + input_modality_dict[registration_pathway[-1]][1][0] + '*'))
                # Error check the fixed volume
                if reference_volume == []:
                    print 'Missing ', input_modality_dict[registration_pathway[-1]][1][0], 'in registration pathway', registration_pathway, '. Skipping this step.'
                    continue
                reference_volume = reference_volume[0]

                if not os.path.exists(os.path.join(output_folder, os.path.basename(reference_volume))) or not os.path.exists(os.path.join(output_folder, os.path.basename(output_folder_not_resampled))):
                    copy(reference_volume, os.path.join(output_folder, os.path.basename(reference_volume)))
                    copy(reference_volume, os.path.join(output_folder_not_resampled, os.path.basename(reference_volume)))

                for moving_volume in input_modality_dict[moving_step][1]:

                    if moving_volume in label_volumes:
                        interpolation = 'nn'
                    else:
                        interpolation = 'linear'

                    moving_volume_filename = glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', input_modality_dict[moving_step][0], patient_visit[0], patient_visit[1], '*' + moving_volume + '*'))
                    if moving_volume_filename == []:
                        print 'Missing ', moving_volume, 'in registration pathway', registration_pathway, '. Skipping this step.'
                        continue        
                    moving_volume_filename = moving_volume_filename[0]

                    print itk_transform_2_numpy(transform_list[0])
                    print get_nifti_affine(moving_volume_filename)
                    print compose_affines(itk_transform_2_numpy(transform_list[0]), get_nifti_affine(moving_volume_filename))

                    moving_suffix, fixed_suffix = get_file_suffixes(moving_volume_filename, reference_volume, visit_code)
                    input_transform = os.path.join(output_folder_transform, visit_code + moving_suffix + '_r_' + fixed_suffix +'.txt')
                    save_affine(final_transform, input_transform)
                    output_volume = os.path.join(output_folder_not_resampled, visit_code + moving_suffix + '_r_' + fixed_suffix +'_o.nii.gz')
                    output_volume_resampled = os.path.join(output_folder, visit_code + moving_suffix + '_r_' + fixed_suffix +'.nii.gz')

                    if not os.path.exists(output_volume_resampled):
                        resample(moving_volume_filename, output_volume_resampled, input_transform=input_transform, reference_volume=reference_volume, command='/opt/Slicer-4.5.0-1-linux-amd64/Slicer', interpolation=interpolation)

                    if not os.path.exists(output_volume):
                        output_affine = compose_affines(final_transform, get_nifti_affine(moving_volume_filename))
                        set_nifti_affine(moving_volume_filename, output_affine, output_filepath=output_volume)

                # # Once all the transforms are accumulated, register for real.
                # if reg_idx == len(registration_pathway) - 2:

                #     for moving_volume in moving_volumes:
                #         print moving_volume
                #         if moving_volume != leader_moving_volume and moving_volume in input_modality_dict[moving_step][1]:
                #             moving_suffix, fixed_suffix = get_file_suffixes(moving_volume, fixed_volume, visit_code)
                #             output_volume = os.path.join(output_folder_not_resampled, visit_code + moving_suffix + '_r_' + fixed_suffix +'.txt')
                #             output_volume_resampled = os.path.join(output_folder, visit_code + moving_suffix + '_r_' + fixed_suffix +'.nii.gz') 
                #             if reg_idx == len(registration_pathway) - 2:
                #                 registration_volume = output_volume
                #             else:
                #                 registration_volume = output_volume_resampled
                #             resample(moving_volume, registration_volume, input_transform=output_transform, reference_volume=reference_volume, command='/opt/Slicer-4.5.0-1-linux-amd64/Slicer')

                # # Find other volumes that could be moved...
                # not_resampled_volumes = glob.glob(os.path.join(output_folder_not_resampled, '*'))
                # for not_resampled_volume in not_resampled_volumes:
                #     pass

                if not transforms:
                    pass

                if not not_resampled:
                    pass

                if not resampled:
                    pass
                # # Multiple volume case.
                # elif registration_mode == 'multiple':
                #     multiple_moving_steps = input_modality_dict[moving_step]
                #     for single_moving_step in multiple_moving_steps:
                #         moving_volumes = [registration_volume for registration_volume in registration_files if single_moving_step in registration_volume]

                # # Error-checking, no volumes found.
                # if len(moving_volumes) == 0:
                #     print 'Moving volume not found for step', registration_pathway[reg_idx:reg_idx+2], '- aborting this pathway.'
                #     continue

                # # Register first volume.
                # if not os.path.exists(output_transform):
                #     register_volume(moving_volume[0], fixed_volume, output_filename=output_volume_resampled, output_transform=output_transform)
                # else:
                #     # get right command
                #     apply_affine_transform(moving_volume[0], output_transform, output_filename=output_volume_resampled)

                # # If applicable, move over the rest of the volumes.
                # if len(moving_volume) > 1:
                #     for additional_volume in moving_volumes[1:]:
                #         moving_suffix, fixed_suffix = get_file_suffixes(additional_volume, fixed_volume)
                #         output_volume = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + fixed_suffix +'.txt')
                #         output_volume_resampled = os.path.join(output_folder, patient_visit + '-' + moving_suffix + '_r_' + fixed_suffix +'.nii.gz')
                #         apply_affine_transform(moving_volume[0], output_transform, output_filename=output_volume_resampled)

        # fd = dg


    # # Useful list to have for future steps.
    # all_modality_list = []
    # for key in input_modality_dict:
    #     all_modality_list += input_modality_dict[key]

    # # Folders to search in base_directory/ANALYSIS
    # modality_folder = ['ANATOMICAL', 'DSC', 'DTI', 'DCE', 'SUV']

    # # Grab all volumes.
    # for folder in modality_folders:
    #     for modality in input_modality_dict[folder]:
    #         folder_niftis = grab_files_recursive(os.path.join(base_directory, study_name, 'ANALYSIS', folder), '*.nii*')

    # # Grab a list of available patient data from the anatomical folder.
    # # TODO: make a more robust method to do these calculations.
    # patient_visit_data = {}
    # for folder in glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', 'ANATOMICAL', '*/')):
    #     patient_num = os.path.basename(folder)
    #     for subfolder in glob.glob(os.path.join(folder, '*/')):
    #         visit_num = os.path.basename(subfolder)
    #         patient_visit_data[patient_num + '-' + visit_num] = glob.glob(os.path.join(subfolder, '*.nii*'))

    # return

def get_file_suffixes(moving_volume, fixed_volume, file_prefix):

    # TODO: Check if file_utils can do this.

    moving_suffix = str.split(os.path.basename(moving_volume), file_prefix)[-1]
    moving_suffix = str.split(moving_suffix, '.')[0]
    fixed_suffix = str.split(os.path.basename(fixed_volume), file_prefix)[-1]
    fixed_suffix = str.split(fixed_suffix, '.')[0]
    
    return moving_suffix, fixed_suffix

def run_test():
    pass

if __name__ == '__main__':
    run_test()