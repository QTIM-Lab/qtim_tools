""" This function is meant to generate ROI-based statistics for QTIM
    volumes.
"""

import numpy as np
import nipype.interfaces.io as nio
import glob
import os
import subprocess
import dicom
import shutil
import sys
import re

from ..qtim_preprocessing.motion_correction import motion_correction
from ..qtim_preprocessing.skull_strip import skull_strip
from ..qtim_utilities.file_util import nifti_splitext
from ..qtim_dti.fit_dti import run_dtifit

def qtim_dti_conversion(study_name, base_directory, output_modalities=[], overwrite=False):

    """ This script is meant for members of the QTIM lab at MGH. It takes in one of our study names,
        a text-identifier for a label volume (e.g. "FLAIR-label"), and an output file, and computes
        a list of label statistics for every file in the "COREGISTRATION" folder. This function can
        also be called with the command-line utility "qtim label_statistics".

        The following statistics are currently computed: mean, min, max, median, range, standard deviation,
        variance, energy, entropy, kurtosis, skewness, COV

        TODO: Make this function more customizable.

        Parameters
        ----------
        study_name: str
            A QTIM study name code, usually three letters.
        base_directory: str
            The full path to the directory from which to search for studies. The study directory
            should be contained in this directory.
        output_modalities: str or list of str
            All of the parameter maps to be outputted by the DTI conversion process.

    """

    # NiPype is not very necessary here, but I want to get used to it. DataGrabber is a utility for
    # for recursively getting files that match a pattern.
    study_files = nio.DataGrabber()
    study_files.inputs.base_directory = base_directory
    study_files.inputs.template = os.path.join(study_name, 'RAWDATA', study_name + '*', 'VISIT_*', 'ep2d*/', '*/')
    study_files.inputs.sort_filelist = True
    results = study_files.run().outputs.outfiles

    # Because of inconsistent naming within QTIM, there are many directories we need to filter out.
    # I create an original_volumes list to hopefully contain only the raw DTI data.
    filter_out_list = ['DFC', 'ADC', 'FA', 'TENSOR', 'TRACEW', 'saraadj']
    original_volumes = []
    for output in results:
        if any(filter_out in output for filter_out in filter_out_list):
            continue
        original_volumes += [output]

    # Each volume is iterated through...
    for volume in original_volumes:

        # Find the output folder
        # TODO: Do this with nipype DataSink instead.
        # Also TODO: Make this easier to change in case directory structure changes.
        split_path = os.path.normpath(volume).split(os.sep)
        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'DTI', split_path[-4], split_path[-3], 'NEW_DTI')

        # Create or delete/create the output folder.
        # if os.path.exists(output_folder):
            # shutil.rmtree(output_folder)    
        # os.mkdir(output_folder)

        # Convert from DICOM
        output_file_prefix = os.path.join(output_folder, split_path[-4] + '-' + split_path[-3]) + '-DTI_diff'
        
        print split_path

        # Don't overwrite if possible.
        if not overwrite and os.path.exists(output_file_prefix + '.bval') and os.path.exists(output_file_prefix + '.bvec') and os.path.exists(output_file_prefix + '.nii.gz'):
            bval, bvec, diff = [output_file_prefix + '.bval', output_file_prefix + '.bvec',output_file_prefix + '.nii.gz']
        else:
            bval, bvec, diff = convert_DTI_nifti(volume, output_folder, output_file_prefix)

        print [bval, bvec, diff]

        # Motion Correction
        output_motion_file = nifti_splitext(diff)[0] + '_mc' + nifti_splitext(diff)[-1]
        print output_motion_file
        if not overwrite and os.path.exists(output_motion_file):
            pass
        else:
            motion_correction(diff, output_motion_file)

        # 1d_tool.py transpose
        output_bvec_file = os.path.splitext(bvec)[0] + '_t' + os.path.splitext(bvec)[-1]
        run_1dtool(bvec, output_bvec_file)

        # Rotate bvecs
        output_rotated_bvec_file = os.path.splitext(output_bvec_file)[0] + '_rotated' + os.path.splitext(output_bvec_file)[-1]
        input_motion_file = nifti_splitext(diff)[0] + '_mc.ecclog'
        run_fdt_rotate_bvecs(output_bvec_file, output_rotated_bvec_file, input_motion_file)

        # Run Skull-Stripping
        skull_strip_mask = nifti_splitext(output_motion_file)[0] + '_ss_mask' + nifti_splitext(output_motion_file)[-1]
        skull_strip_output = nifti_splitext(output_motion_file)[0] + '_ss' + nifti_splitext(output_motion_file)[-1]
        if not os.path.exists(skull_strip_mask):
            skull_strip(output_motion_file, skull_strip_output, skull_strip_mask, extra_parameters={'fsl_threshold': .1})

        output_prefix = os.path.join(output_folder, split_path[-4] + '-' + split_path[-3] + '-DTI')
        run_dtifit(output_motion_file, output_rotated_bvec_file, bval, input_mask=skull_strip_mask, output_fileprefix=output_prefix)


    return

def run_fdt_rotate_bvecs(input_bvec, output_bvec, input_motion):

    script_location = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','external','fdt_rotate_bvecs.sh'))

    print ' '.join([script_location, input_bvec, output_bvec, input_motion])

    subprocess.call([script_location, input_bvec, output_bvec, input_motion])

def run_1dtool(input_bvec, output_bvec):

    print input_bvec, output_bvec
    print ' '.join(['1d_tool.py', '-infile', input_bvec, '-transpose', '-write', output_bvec])
    subprocess.call(['1d_tool.py', '-infile', input_bvec, '-transpose', '-write', output_bvec])

    return

def convert_DTI_nifti(volume, output_folder, output_file_prefix):

    # Grab all dicom files in a directory. Try to find a usable reference DICOM among them.
    dicom_list = glob.glob(os.path.join(volume, '*'))
    reference_dicom = []
    for dicom_file in dicom_list:
        try:
            reference_dicom = dicom.read_file(dicom_file)
            break
        except:
            continue       

    # If we can't find any DICOMs in these folders, skip this volume.
    if not reference_dicom:
        print 'No DICOMs found! Skipping folder... ', volume
        return False

    # Attempt to mine sequence, protocol, bvalue data from the reference DICOM.
    # The following code is courtesy Karl's group.
    sequence, protocol, bval = [], '', ''

    #####GENERAL SEQUENCE####
    try:
        sequence.append(reference_dicom[0x18,0x24].value)
    except:
        pass
    try:
        sequence.append(reference_dicom[0x18,0x20].value)
    except:
        pass

    #####GENERAL PROTOCOL#####
    try:
        protocol = reference_dicom[0x18,0x1030].value
    except:
        pass

    try:
        bval = reference_dicom[0x18,0x9087].value
    except:
        pass

    ####SIEMENS Specific Private Tags####
    if reference_dicom.Manufacturer == 'SIEMENS':
        if bval == '':
            try:
                bval = reference_dicom[0x19,0x100C].value
            except:
                pass

    #####GE Specific Private Tags####### 
    elif reference_dicom.Manufacturer == 'GE MEDICAL SYSTEMS':
        try:
            sequence.append([0x19,0x109C].value)
        except:
            pass
        if bval == '':
            try:
                bval = reference_dicom[0x43,0x1039].value
            except:
                pass

    # Check if the sequence/protocol/bvalue information is as expected. If so, convert nifti.
    DTI = check_DTI(sequence, str(protocol), str(bval))
    if DTI == True:
        subprocess.call(['dcm2nii', '-d', 'N', '-i', 'N', '-p', 'N', '-o', output_folder, volume])

    # Rename Output Files. TODO: Use split_file constructions less.
    output_files = sorted(glob.glob(os.path.join(output_folder, '*')))
    renamed_output_files = []

    for output_file in output_files:
        renamed_file = os.path.join(output_folder, output_file_prefix + '.' + '.'.join(str.split(os.path.basename(output_file), '.')[1:]))
        renamed_output_files += [renamed_file]
        shutil.move(output_file, renamed_file)

    return renamed_output_files

def check_DTI(sequence, protocol, bval):

    """ This script is courtesy Karl's group. Documentation to come.
    """

    DTI = False

    # if none of the elements of sequence match any of the given values
    if not any([x in ["tof", "fl3d", "memp", "fse", "grass", "3-Plane", "gre" ] for x in sequence]):
        if re.search('(ep2)|b|(ep_)', str(sequence), re.IGNORECASE):
            DTI = True
        elif re.search('(1000)|(directional)',bval, re.IGNORECASE):
            DTI = True
        elif re.search('dif', protocol, re.IGNORECASE):
            DTI = True
        else:
            pass

    return DTI

def run_test():
    return

if __name__ == '__main__':
    run_test()