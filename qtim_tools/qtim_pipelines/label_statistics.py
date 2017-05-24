""" This function is meant to generate ROI-based statistics for QTIM
    volumes.
"""

import numpy as np
import nipype.interfaces.io as nio
import glob
import os

from ..qtim_features.statistics import qtim_statistic
from ..qtim_utilities.text_util import save_numpy_2_csv

def qtim_study_statistics(study_name, label_file, output_csv, base_directory):

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
            a warning will be raised and the first alphabetical label will be used.
        output_csv: str
            The full path to the output csv file. This file will contain label statistics
        base_directory: str
            The full path to the directory from which to search for studies. The study directory
            should be contained in this directory.

    """

    # These are all the features currently available in QTIM.
    features_calculated = ['mean','min','max','median','range','standard_deviation','variance','energy','entropy','kurtosis','skewness','COV']

    # NiPype is not very necessary here, but I want to get used to it. DataGrabber is a utility for
    # for recursively getting files that match a pattern.
    study_files = nio.DataGrabber()
    study_files.inputs.base_directory = base_directory
    study_files.inputs.template = os.path.join(study_name, 'ANALYSIS', 'COREGISTRATION', study_name + '*', 'VISIT_*', '*.ni*')
    study_files.inputs.sort_filelist = True
    results = study_files.run().outputs.outfiles

    # Toss out labels from calculated statistics.
    outputs_without_labels = [study_file for study_file in results if label_file not in study_file]

    # Create the CSV output array.
    output_numpy = np.full((1+len(outputs_without_labels), 1+len(features_calculated)), 'NA', dtype=object)

    for return_idx, return_file in enumerate(outputs_without_labels):

        # For each file, make a row and check if the label file exists.
        directory = os.path.dirname(return_file)
        visit_label = sorted(glob.glob(os.path.join(directory, '*' + label_file + '*')))
        output_numpy[return_idx+1, 0] = return_file

        if visit_label:

            # If multiple labels returned, give a warning and use the first option returned.
            if len(visit_label) != 1:
                print 'WARNING! Multiple labels found. Going with... ' + visit_label[0]

            # Use qtim's statistics package to calculate label statistics.
            print return_file
            output_numpy[return_idx+1, 1:] = qtim_statistic(return_file, features_calculated, visit_label[0])

        else:
            print 'Warning! No label found in the same directory as... ', return_file

    # Create CSV headers.
    output_numpy[0,:] = ['filename'] + features_calculated

    # Save output
    save_numpy_2_csv(output_numpy, output_csv)

    return

def run_test():
    return

if __name__ == '__main__':
    run_test()