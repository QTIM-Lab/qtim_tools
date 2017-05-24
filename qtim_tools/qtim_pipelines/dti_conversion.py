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

def qtim_dti_conversion(study_name, base_directory='/qtim2/users/data/'):

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
        label_file: str
            A phrase that specifically indicates the label to generate statistics from.
            For example, 'FLAIR-label', 'T1POST-label'. If multiple labels are returned,
            a warning will be raised an only one will be calculated.
        output_csv: str
            The full path to the output csv file. This file will contain label statistics
        base_directory: str
            The full path to the directory from which to search for studies. The study directory
            should be contained in this directory.

    """

    # NiPype is not very necessary here, but I want to get used to it.
    study_files = nio.DataGrabber()
    study_files.inputs.base_directory = base_directory
    study_files.inputs.template = os.path.join(study_name, 'RAWDATA', study_name + '*', 'VISIT_*', 'ep2d*/', '*/')
    study_files.inputs.sort_filelist = True
    results = study_files.run().outputs.outfiles

    precomputed_list = ['DFC', 'ADC', 'FA', 'TENSOR', 'TRACEW', 'saraadj']
    original_volumes = []

    for output in results:
        if any(phrase in output for phrase in precomputed_list):
            continue
        original_volumes += [output]

    for volume in original_volumes:

        split_path = os.path.normpath(volume).split(os.sep)

        output_folder = os.path.join(base_directory, study_name, 'ANALYSIS', 'DTI')

        print split_path

    # output_numpy = np.full((1+len(outputs_without_labels), 1+len(features_calculated)), 'NA', dtype=object)

    # for return_idx, return_file in enumerate(outputs_without_labels):

    #     directory = os.path.dirname(return_file)
    #     visit_label = glob.glob(os.path.join(directory, '*' + label_file + '*'))

    #     output_numpy[return_idx+1, 0] = return_file

    #     if visit_label:

    #         if len(visit_label) != 1:
    #             print 'WARNING! Multiple labels found. Going with... ' + visit_label[0]

    #         print return_file

    #         output_numpy[return_idx+1, 1:] = qtim_statistic(return_file, features_calculated, visit_label[0])

    # output_numpy[0,:] = ['filename'] + features_calculated

    # save_numpy_2_csv(output_numpy, output_csv)

    return

def check_DTI(sequence, protocol, bval):
    DTI = False

    # if none of the elements of sequence match any of the given values
    if False not in [x not in ["tof", "fl3d", "memp", "fse", "grass", "3-Plane", "gre" ] for x in sequence]:
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

# root = '/qtim2/users/data/FMS/RAWDATA/FMS_01/VISIT_01/ep2d_diff_SliceAcc3_PAT2/'
# nifti = '/qtim/users/jcardo/Karl_example/'

# for sub in os.listdir(root):
#     if os.path.isdir(root+sub):
#     if os.path.exists(nifti+sub) == False:
#         os.mkdir(nifti+sub)
#             #for run in os.listdir(root+sub):
#                     #if os.path.isdir(root+sub+'/'+run) == True:
#                             if any(mm.endswith('.dcm') for mm in os.listdir(rooti+sub)):
#                 if os.path.exists(nifti+sub+'/'+run) == False:
#                     os.mkdir(nifti+sub+'/'+run)
#                                     sequence = []
#                                     protocol, bval = '',''
#                                     f = os.listdir(root+sub+'/'+run)[0]
#                                     try:
#                                             ds = dicom.read_file(root+sub+'/'+run+'/'+f)
#                                     except:
#                                             f = os.listdir(root+sub+'/'+run)[1]
#                                             ds = dicom.read_file(root+sub+'/'+run+'/'+f)

#                                     #####GENERAL SEQUENCE####
#                                     try:
#                                             sequence.append(ds[0x18,0x24].value)
#                                     except:
#                                             pass
#                                     try:
#                                             sequence.append(ds[0x18,0x20].value)
#                                     except:
#                                             pass

#                                     #####GENERAL PROTOCOL#####
#                                     try:
#                                             protocol = ds[0x18,0x1030].value
#                                     except:
#                                             pass

#                                     try:
#                                             bval = ds[0x18,0x9087].value
#                                     except:
#                                             pass

#                                     ####SIEMENS Specific Private Tags####
#                                     if ds.Manufacturer == 'SIEMENS':
#                                                     if bval == '':
#                                                             try:
#                                                                     bval = ds[0x19,0x100C].value
#                                                             except:
#                                                                     pass

#                                     #####GE Specific Private Tags####### 
#                                     elif ds.Manufacturer == 'GE MEDICAL SYSTEMS':
#                                             try:
#                                                     sequence.append([0x19,0x109C].value)
#                                             except:
#                                                     pass
#                                             if bval == '':
#                                                     try:
#                                                             bval = ds[0x43,0x1039].value
#                                                     except:
#                                                             pass
#                                     else:
#                                             pass

#                                     DTI = checkIfDTI(sequence, str(protocol), str(bval))
#                                     if DTI == True:
#                                             subprocess.call(['dcm2nii', '-d', 'N', '-i', 'N', '-p', 'N', '-o', nifti+sub+'/'+run, root+sub+'/'+run])
