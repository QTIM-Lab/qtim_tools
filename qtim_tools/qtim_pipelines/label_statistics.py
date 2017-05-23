""" This function is meant to generate ROI-based statistics for QTIM
    volumes.
"""

import numpy as np
import nipype.interfaces.io as nio
import glob
import os

from ..qtim_features.statistics import qtim_statistic
from ..qtim_utilities.text_util import save_numpy_2_csv

def qtim_study_statistics(study_name, label_file, output_csv, base_directory='/qtim2/users/data/'):

    features_calculated = ['mean','min','max','median','range','standard_deviation','variance','energy','entropy','kurtosis','skewness','COV']

    # NiPype is not very necessary here, but I want to get used to it.
    study_files = nio.DataGrabber()
    study_files.inputs.base_directory = base_directory
    study_files.inputs.template = os.path.join(study_name, 'ANALYSIS', 'COREGISTRATION', study_name + '*', 'VISIT_*', '*.ni*')
    study_files.inputs.sort_filelist = True
    results = study_files.run().outputs.outfiles

    outputs_without_labels = [study_file for study_file in results if label_file not in study_file]
    output_numpy = np.full((1+len(outputs_without_labels), 1+len(features_calculated)), 'NA', dtype=object)

    for return_idx, return_file in enumerate(outputs_without_labels):

        directory = os.path.dirname(return_file)
        visit_label = glob.glob(os.path.join(directory, '*' + label_file + '*'))

        output_numpy[return_idx+1, 0] = return_file

        if visit_label:

            if len(visit_label) != 1:
                print 'WARNING! Multiple labels found. Going with... ' + visit_label[0]

            print return_file

            output_numpy[return_idx+1, 1:] = qtim_statistic(return_file, features_calculated, visit_label[0])

    output_numpy[0,:] = ['filename'] + features_calculated

    save_numpy_2_csv(output_numpy, output_csv)

    return

def run_test():
    return

if __name__ == '__main__':
    run_test()