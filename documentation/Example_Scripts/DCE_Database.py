import numpy as np
import os
import glob
import csv

from shutil import copy

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

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

def Save_Directory_Statistics(input_directory, ROI_directory, mask=False, mask_suffix='_mask'):

    """ Save ROI statistics into a giant csv file.
    """

    file_database = glob.glob(os.path.join(input_directory, '*.nii*'))

    output_headers = ['filename','mean','median','std','min','max','total_voxels','removed_values', 'removed_percent']

    output_csv = np.zeros((1+len(file_database), len(output_headers)),dtype=object)
    output_csv[0,:] = output_headers

    ROI_dict = {}

    for ROI in glob.glob(os.path.join(ROI_directory, '*.nii*')):
        ROI_dict[os.path.basename(os.path.normpath(ROI))[0:15]] = convert_input_2_numpy(ROI)

    with open('DCE_Assay' + '.csv', 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')

        for row_idx, filename in enumerate(file_database):

            data_array = convert_input_2_numpy(filename)
            patient_visit_code = os.path.basename(os.path.normpath(filename))[0:15]
            roi_array = ROI_dict[patient_visit_code]

            masked_data_array_invalid = np.ma.masked_where(data_array <= 0, data_array)
            # masked_data_array_ROI = np.ma.masked_where(roi_array <= 0, data_array)

            ROI_values = [np.ma.mean(masked_data_array_invalid), np.ma.median(masked_data_array_invalid), np.ma.min(masked_data_array_invalid), np.ma.max(masked_data_array_invalid), np.ma.std(masked_data_array_invalid),(roi_array > 0).sum(), ((data_array <= 0) & (roi_array > 0)).sum(), float(((data_array <= 0) & (roi_array > 0)).sum()) / float((roi_array > 0).sum())]

            print ROI_values

            output_csv[row_idx] = [filename] + ROI_values

            csvfile.writerow(output_csv[row_idx])

    return

if __name__ == '__main__':

    data_directory = '/home/abeers/Data/DCE_Package/Test_Results/Echo1'
    NHX_directory = '/qtim2/users/data/NHX/ANALYSIS/DCE/'
    CED_directory = '/qtim/users/data/CED/ANALYSIS/DCE/PREPARATION_FILES/'
    ROI_directory = '/home/abeers/Data/DCE_Package/Test_Results/ROIs'

    # Copy_SameAIF_Visit1_Tumors(data_directory)
    Create_ROI_Directory(CED_directory, NHX_directory, ROI_directory)
    Save_Directory_Statistics(data_directory, ROI_directory)

    pass