from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.threshold import crop_with_mask
from ..qtim_preprocessing.resample import resample
from ..qtim_preprocessing.normalization import zero_mean_unit_variance
from ..qtim_preprocessing.bias_correction import bias_correction
from ..qtim_utilities.file_util import nifti_splitext

def deep_learning_preprocess(study_name, base_directory, skull_strip_label='T2', skip_modalities=[]):

    """ This script is meant for members of the QTIM lab at MGH. It takes in one of our study names, finds the
        COREGISTRATION folder, presumed to have been created in an earlier part of the pipeline, and applies
        the following pre-processing steps to our data to make it ready for input into deep-learning algorithms:
        
        1) N4 Bias Correction
        2) Isotropic Resampling
        3) Skull-Stripping
        4) Zero Mean Normalization

        This is a work in progress, and may not be appropriate for every deep-learning task. For example, data
        completion tasks may not benefit from zero-mean normalization.

        Parameters
        ----------
        study_name: str
            A QTIM study name code, usually three letters.
        base_directory: str
            The full path to the directory from which to search for studies. The study directory
            should be contained in this directory.
        skull_strip_label: str
            An text identifier (e.g. "FLAIR") for which volume should be skull-stripped. All other volumes will be stripped according to this identifier.
        skip_modalities: str or list of str
            Any modalities that should not be processed.

    """

    # NiPype is not very necessary here, but I want to get used to it. DataGrabber is a utility for
    # for recursively getting files that match a pattern.
    study_files = nio.DataGrabber()
    study_files.inputs.base_directory = base_directory
    study_files.inputs.template = os.path.join(study_name, 'ANALYSIS', 'COREGISTRATION', study_name + '*', 'VISIT_*', '*.nii.gz')
    study_files.inputs.sort_filelist = True
    results = study_files.run().outputs.outfiles

    # Remove modality codes that will not be processed for deep-learning.
    dl_volumes = []
    for output in results:
        if any(modality in output for modality in skip_modalities):
            continue
        dl_volumes += [output]

    # We prefer the same skull-stripping in each case. Thus, we first iterate through skull-strip volumes.
    skull_strip_volumes = [volume for volume in dl_volumes if skull_strip_label in volume]

    # Strip the chosen volume..
    for skull_strip_volume in skull_strip_volumes:

        # Find the output folder
        # TODO: Do this with nipype DataSink instead.
        # Also TODO: Make this easier to change in case directory structure changes.
        split_path = os.path.normpath(volume).split(os.sep)
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'DEEPLEARNING', split_path[-4], split_path[-3])

        # Make the output directory if it does not exist.
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # If skull-stripping has not yet been performed, perform it.
        skull_strip_mask = os.path.join(output_folder, split_path[-4] + '-' split_path[-3] + '-' + 'SKULL_STRIP_MASK.nii.gz')
        skull_strip_output = os.path.join(output_folder, nifti_splitext(skull_strip_volume)[0] + '_ss' + nifti_splitext(skull_strip_volume)[-1])
        if not os.path.exists(skull_strip_mask):
            skull_strip(skull_strip_volume, skull_strip_output, skull_strip_mask)

    # Skull-Strip, Resample, and Normalize the rest. 
    for dl_volume in dl_volumes:

        split_path = os.path.normpath(volume).split(os.sep)
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'DEEPLEARNING', split_path[-4], split_path[-3])

        # Make sure a mask was created in the previous step.
        skull_strip_mask = os.path.join(output_folder, split_path[-4] + '-' split_path[-3] + '-' + 'SKULL_STRIP_MASK.nii.gz')
        if not os.path.exists(skull_strip_mask):
            print 'No skull-stripping mask created, skipping this volume!'
            continue

        # Use existing mask to skull-strip if necessary.
        n4_bias_output = os.path.join(output_folder, nifti_splitext(dl_volume)[0] + '_n4' + nifti_splitext(dl_volume)[-1])
        if not os.path.exists(skull_strip_output):
            bias_correction(dl_volume, output_filename=n4_bias_output)

        # Use existing mask to skull-strip if necessary.
        skull_strip_output = os.path.join(output_folder, nifti_splitext(n4_bias_output)[0] + '_ss' + nifti_splitext(n4_bias_output)[-1])
        if not os.path.exists(skull_strip_output):
            crop_with_mask(n4_bias_output, skull_strip_mask, output_filename=skull_strip_output)
        os.remove(n4_bias_output)

        # Resample and remove previous file.
        resample_output = os.path.join(output_folder, nifti_splitext(skull_strip_output)[0] + '_iso' + nifti_splitext(skull_strip_output)[-1])
        if not os.path.exists(skull_strip_output):
            resample(skull_strip_output, output_filename=resample_output)
        os.remove(skull_strip_output)

        # Mean normalize and remove previous file.
        normalize_output = os.path.join(output_folder, nifti_splitext(dl_volume)[0] + '_DL' + nifti_splitext(dl_volume)[-1])
        if not os.path.exists(skull_strip_output):
            zero_mean_unit_variance(resample_output, output_filename=normalize_output)
        os.remove(resample_output)

    return

def run_test():
    pass

if __name__ == '__main__':
    run_test()