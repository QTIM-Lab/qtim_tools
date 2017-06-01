""" This function is meant to generate ROI-based statistics for QTIM
    volumes.
"""

import numpy as np
import nipype.interfaces.io as nio
import glob
import os

from ..qtim_features.statistics import qtim_statistic
from ..qtim_utilities.text_util import save_numpy_2_csv
from ..qtim_utilities.format_util import convert_input_2_numpy
from ..qtim_utilities.file_util import grab_files_recursive

def qtim_study_statistics(study_name, label_file, base_directory, output_csv=None, label_mode='combined'):

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

    base_directory = os.path.abspath(base_directory)

    # Defaults
    if output_csv is None:
        output_csv = os.path.join(base_directory, study_name, 'ANALYSIS', 'STATISTICS', study_name + '_' + label_file + '_statistics.csv')
    if label_mode is None:
        label_mode = 'combined'

    # These are all the features currently available in QTIM.
    features_calculated = ['mean','min','max','median','range','standard_deviation','variance','energy','entropy','kurtosis','skewness','COV']

    # Column titles. Replace these lines with a config file later...
    modalities = ['T1Post_r_T2', 'T1Pre_r_T2', 'MPRAGE_POST_r_T2', 'MPRAGE_Pre_r_T2', 'FLAIR_r_T2', 'T2SPACE', 'SUV_r_T2', 'DSC_GE_r_T2', 'DSC_GE_CBV_r_T2', 'DSC_GE_rBF_r_T2', 'DSC_SE_r_T2', 'DSC_SE_CBV_r_T2', 'DSC_SE_rBF_r_T2', 'DCE_r_T2', 'DCE_ktrans_r_T2']
    differences = [('T1Post_r_T2', 'T1Pre_r_T2'), ('MPRAGE_POST_r_T2', 'MPRAGE_Pre_r_T2')]

    # Exlcusion phrases
    label_exclusions = [label_file] + ['label']

    # NiPype is not very necessary here, but I want to get used to it. DataGrabber is a utility for
    # for recursively getting files that match a pattern.
    all_patients = sorted(glob.glob(os.path.join(base_directory, study_name, 'ANALYSIS', 'COREGISTRATION', '*/')))

    # all_coregistred_files = grab_files_recursive(os.path.join(study_name, 'ANALYSIS', 'COREGISTRATION'), '*.nii*')

    # Create the CSV output array.
    output_numpy = np.array(['filename'] + modalities + [modality1 + '_minus_' + modality2 for modality1, modality2 in differences])

    for patient in all_patients:

        all_visits = sorted(glob.glob(os.path.join(patient, '*/')))

        for visit in all_visits:

            print 'Calculating statistics for... ', visit

            visit_label = sorted(glob.glob(os.path.join(visit, '*' + label_file + '*')))
            print visit_label

            if visit_label:

                # If multiple labels returned, give a warning and use the first option returned.
                if len(visit_label) != 1:
                    print 'WARNING! Multiple labels found. Going with... ' + visit_label[0]
                visit_label = convert_input_2_numpy(visit_label[0])

                if label_mode == 'combined':
                    label_list = ['']
                elif label_mode != 'separate': 
                    print 'Provided label_mode,', label_mode, 'is not an available option. Going with \'combined\'.'
                    label_list = ['']
                else:
                    label_list = ['_label-' + str(int(label_num)) for label_num in np.unique(visit_label)[1:]]

                print label_list

                for label_index in label_list:

                    new_row = ['']*len(['filename'] + modalities + [modality1 + '_minus_' + modality2 for modality1, modality2 in differences])
                    col_number = 1
                    new_row[0] = str.split(visit, os.sep)[-3] + '-' + str.split(visit, os.sep)[-2] + label_index

                    for modality in modalities:

                        modality_file = glob.glob(os.path.join(visit, '*' + modality + '*'))

                        if len(modality_file) == 0:
                            print 'Modality file for modality label', modality, 'not found, skipping this modality'
                            new_row[col_number] = ''
                        else:
                            if len(modality_file) > 1:
                                print 'Found multiple files for modality label', modality, '! Choosing the first one found,', modality_file[0]
                            else:
                                print 'Calculating statistic for modality label', modality
                            # try:
                            if label_index != '':
                                new_row[col_number] = qtim_statistic(modality_file[0], ['median'], visit_label, return_label=int(label_index[-1]))[0]
                            else:
                                new_row[col_number] = qtim_statistic(modality_file[0], ['median'], visit_label)[0]
                            # except:
                                # print 'Error calculating statistic for modality label', modality
                                # new_row[col_number] = ''
                            if new_row[col_number] == 'nan':
                                print 'Error calculating statistic for modality label', modality
                                new_row[col_number] = ''

                        col_number += 1

                    for difference in differences:

                        try:
                            new_row[col_number] = float(new_row[1+modalities.index(difference[0])]) - float(new_row[1+modalities.index(difference[1])])
                        except:
                            print "Error occured while calculating difference for", difference, '. Skipping this statistic.'
                            new_row[col_number] = ''

                        col_number += 1

                    print new_row
                    output_numpy = np.vstack((output_numpy, new_row))

            else:
                print 'Warning! No label found in the same directory as... ', visit_label

        output_numpy = np.vstack((output_numpy, ['']*len(['filename'] + modalities + [modality1 + '_minus_' + modality2 for modality1, modality2 in differences])))

    save_numpy_2_csv(output_numpy, output_csv)

    return

def run_test():
    return

if __name__ == '__main__':
    run_test()