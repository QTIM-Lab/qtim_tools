import numpy as np
import os
import glob
import csv

from shutil import copy

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

# Categorical
BLUR = ['blur_' + x for x in ['0','0.2','0.8','1.2']]
PCA = ['pca_' + x for x in ['0','1','2','3','4']]
THRESHOLD = ['threshold_' + x for x in ['-1', '0.01']]
ALGORITHM = ['simplex', 'lm']
INTEGRATION = ['recursive','conv']


# Binary
AIF = ['autoAIF']
SAME_AIF = ['sameAIF']
T1MAP = ['t1map']

# All
ALL_VARS = [BLUR, PCA, THRESHOLD, ALGORITHM, INTEGRATION, AIF, SAME_AIF, T1MAP]

def Copy_SameAIF_Visit1_Tumors(input_directory):

    """ For one vs. all comparisons, it will be easier to code if we have duplicate
        visit 1 entries for the "SameAIF" mode. Because SameAIF uses the AIF from visit
        1 in both entries, visit 1 should be unchanged with this option.
    """

    visit_1_file_database = glob.glob(os.path.join(input_directory, '*VISIT_01_autoAIF*.nii*'))

    for filename in visit_1_file_database:
        old_filename = str.split(filename, 'VISIT_01')[0] + '_sameAIF_' + str.split(filename, 'VISIT_01')[-1]
        new_filename = str.split(filename, 'VISIT_01')[0] + 'VISIT_01_sameAIF_' + str.split(filename, 'VISIT_01')[-1]
        copy(filename, new_filename)
        os.remove(old_filename)

def Rename_LM_Files(data_directory):

    lm_file_database = glob.glob(os.path.join(data_directory, '*lm*lm*.nii*'))

    for filename in lm_file_database:

        # split_file = str.split(filename, '_lm_')
        # print split_file
        # new_filename = split_file[0] + '_' + split_file[1] + '_lm_' + split_file[2]
        print filename

        # copy(filename, new_filename)
        os.remove(filename)

    return

def Delete_Extra_Files(data_directory):

    file_database = glob.glob(os.path.join(data_directory, '*.nii*'))

    for files in file_database:
        if '__lm__' not in files and 'simplex' not in files:
            # print os.path.basename(os.path.normpath(files))
            os.remove(files)

    return

def Create_ROI_Directory(CED_directory, NHX_directory, output_folder):

    """ Move all ROIs for the DCE script from their idiosyncratic locations
        to a set folder.
    """

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # print 

    NHX_dirs = glob.glob(os.path.join(NHX_directory, 'NHX*/'))
    CED_dirs = glob.glob(os.path.join(CED_directory, 'CED*/'))

    visits = ['VISIT_01', 'VISIT_02']

    for visit in visits:

        for NHX_dir in NHX_dirs:

            ROI = os.path.join(NHX_dir, visit, 'ROISTATS', 'T1AxialPost', 'rT1AxialPostROI.nii')
            output_path = os.path.join(output_folder, os.path.basename(os.path.normpath(NHX_dir)) + '_' + visit + '_ROI.nii')
            if os.path.exists(ROI):
                print output_path
                copy(ROI, output_path)

        for CED_dir in CED_dirs:

            ROI = os.path.join(CED_dir, visit, 'ROISTATS', 'T1AxialPost', 'rT1AxialPostROI.nii')
            output_path = os.path.join(output_folder, os.path.basename(os.path.normpath(CED_dir)) + '_' + visit + '_ROI.nii')
            if os.path.exists(ROI):
                print output_path
                copy(ROI, output_path)

def Save_Directory_Statistics(input_directory, ROI_directory, output_csv, mask=False, mask_suffix='_mask'):

    """ Save ROI statistics into a giant csv file.
    """

    file_database = glob.glob(os.path.join(input_directory, '*.nii*'))

    output_headers = ['filename','mean','median','std','min','max','total_voxels','removed_values', 'removed_percent']

    output_data = np.zeros((1+len(file_database), len(output_headers)),dtype=object)
    output_data[0,:] = output_headers

    ROI_dict = {}

    for ROI in glob.glob(os.path.join(ROI_directory, '*.nii*')):
        ROI_dict[os.path.basename(os.path.normpath(ROI))[0:15]] = convert_input_2_numpy(ROI)

    with open(output_csv, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        csvfile.writerow(output_data[0,:])

        for row_idx, filename in enumerate(file_database):

            data_array = convert_input_2_numpy(filename)
            patient_visit_code = os.path.basename(os.path.normpath(filename))[0:15]
            roi_array = ROI_dict[patient_visit_code]

            masked_data_array_invalid = np.ma.masked_where(data_array <= 0, data_array)
            # masked_data_array_ROI = np.ma.masked_where(roi_array <= 0, data_array)

            ROI_values = [np.ma.mean(masked_data_array_invalid), np.ma.median(masked_data_array_invalid), np.ma.min(masked_data_array_invalid), np.ma.max(masked_data_array_invalid), np.ma.std(masked_data_array_invalid),(roi_array > 0).sum(), ((data_array <= 0) & (roi_array > 0)).sum(), float(((data_array <= 0) & (roi_array > 0)).sum()) / float((roi_array > 0).sum())]

            print ROI_values

            output_data[row_idx+1] = [filename] + ROI_values

            csvfile.writerow(output_data[row_idx+1])

    return

def Reshape_Statisticts_Worksheet(input_csv, output_csv, ROI_directory):

    """ TBD
    """

    return

def Paired_Visits_Worksheet(input_csv, output_csv, grab_column=2):

    input_data = np.genfromtxt(input_csv, delimiter=',', dtype=object, skip_header=1)

    visit_1_list = [x for x in input_data[:,0] if 'VISIT_01' in x]

    output_data = np.zeros((len(visit_1_list)+1, 3), dtype=object)
    output_data[0,:] = ['method_code', 'visit_1', 'visit_2']

    with open(output_csv, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        csvfile.writerow(output_data[0,:])

        for visit_idx, visit in enumerate(visit_1_list):

            split_visit = str.split(visit, 'VISIT_01')
            new_visit = split_visit[0] + 'VISIT_02' + split_visit[1]

            if new_visit in input_data[:,0]:
                
                print np.where(input_data == visit)[0][0]

                output_data[visit_idx+1, 0] = visit
                output_data[visit_idx+1, 1] = input_data[np.where(input_data == visit)[0][0], grab_column]
                output_data[visit_idx+1, 2] = input_data[np.where(input_data == new_visit)[0][0], grab_column]

            if output_data[visit_idx, 0] != 0 and output_data[visit_idx, 0] != '0':
                csvfile.writerow(output_data[visit_idx+1,:])

def Coeffecient_of_Variation_Worksheet(input_csv, output_csv):

    input_data = np.genfromtxt(input_csv, delimiter=',', dtype=object, skip_header=1)
    output_data = np.zeros((3000, 5), dtype=object)
    output_data[0,:] = ['method', 'RMS_COV', 'LOG_COV', 'SD_COV', 'n_measurements']

    methods = []

    with open(output_csv, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        csvfile.writerow(output_data[0,:])

        row_idx = 0

        for row in input_data:

            if row[0] == '0' or '--' in row:
                continue

            method = str.split(row[0], '/')[-1][15:]

            # print row[0]
            print method

            if method not in methods:
                # patient_list = np.where(method in input_data)
                patient_list = [method == str.split(x, '/')[-1][15:] for x in input_data[:,0]]
                patient_list = input_data[patient_list, :]
                # print patient_list
                # print 'METHOD', method

                # Root Mean Square Method
                RMS_sum = 0
                LOG_sum = 0
                SD_sum_1 = 0
                SD_sum_2 = 0
                n = 0

                for patient in patient_list:

                    # print patient

                    if '--' in patient:
                        continue

                    data_points = [float(x) for x in patient[1:]]

                    RMS_sum += np.power(abs(data_points[0] - data_points[1]) / np.mean(data_points), 2)
                    LOG_sum += np.power(np.log(data_points[0]) - np.log(data_points[1]), 2)
                    SD_sum_1 += np.power(data_points[0] - data_points[1], 2)
                    SD_sum_2 += np.sum(data_points)
                    n += 1

                RMS_COV = 100 * np.sqrt(RMS_sum / (2*n))
                LOG_COV = 100 * np.exp(np.sqrt(LOG_sum / (2*n)) - 1)
                SD_COV = 100 * np.sqrt(SD_sum_1 / (2*n)) / (SD_sum_2 / (2*n))

                output_data[row_idx+1, :] = [method, RMS_COV, LOG_COV, SD_COV, n]
                # print output_data[row_idx+1, :]

                methods += [method]
                # print methods

                if output_data[row_idx+1, 0] != 0 and output_data[row_idx+1, 0] != '0':
                    csvfile.writerow(output_data[row_idx+1,:])

                row_idx += 1


            else:
                # print 'SKIPPED!!!!'
                continue


    return

if __name__ == '__main__':

    data_directory = '/home/abeers/Data/DCE_Package/Test_Results/Echo1'
    NHX_directory = '/qtim2/users/data/NHX/ANALYSIS/DCE/'
    CED_directory = '/qtim/users/data/CED/ANALYSIS/DCE/PREPARATION_FILES/'
    ROI_directory = '/home/abeers/Data/DCE_Package/Test_Results/ROIs'

    output_csv = 'DCE_Assay.csv'
    reshaped_output_csv = 'DCE_Assay_Split.csv'
    paired_csv = 'DCE_Assay_Visits_Paired.csv'
    paired_reduced_csv = 'DCE_Assay_Visits_Paired_Extremes_Culled.csv'
    cov_csv = 'DCE_Assay_COV.csv'
    cov_reduced_csv = "DCE_Assay_COV_reduce.csv"

    # Rename_LM_Files(data_directory)
    # Copy_SameAIF_Visit1_Tumors(data_directory)
    # Create_ROI_Directory(CED_directory, NHX_directory, ROI_directory)
    # Save_Directory_Statistics(data_directory, ROI_directory, output_csv)
    # Reshape_Statisticts_Worksheet(output_csv, reshaped_output_csv, ROI_directory)
    # Delete_Extra_Files(data_directory)
    # Paired_Visits_Worksheet(output_csv, paired_csv)
    # Coeffecient_of_Variation_Worksheet(paired_csv, cov_csv)
    Coeffecient_of_Variation_Worksheet(paired_reduced_csv, cov_reduced_csv)

    pass