import numpy as np
import os
import glob
import csv


from shutil import copy, move
from sklearn.metrics import r2_score
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.file_util import replace_suffix
from qtim_tools.qtim_utilities.nifti_util import nifti_resave, save_numpy_2_nifti
from collections import defaultdict

# Categorical
BLUR = ['blur_' + x for x in ['0','0.2','0.8','1.2']]
PCA = ['pca_' + x for x in ['0','1','2','3','4']]
THRESHOLD = ['threshold_' + x for x in ['-1', '0.01']]
ALGORITHM = ['simplex', 'lm']
INTEGRATION = ['recursive','conv']
AIF = ['autoAIF', 'popAIF', 'sameAutoAIF']
T1MAP = ['t1map','t1static']

# All
ALL_VARS = [BLUR, PCA, THRESHOLD, ALGORITHM, INTEGRATION, AIF, T1MAP]

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
        # print(split_file)
        # new_filename = split_file[0] + '_' + split_file[1] + '_lm_' + split_file[2]
        print(filename)

        # copy(filename, new_filename)
        # os.remove(filename)

    return

def Delete_Extra_Files(data_directory):

    file_database = glob.glob(os.path.join(data_directory, '*.nii*'))

    for files in file_database:
        if '__lm__' not in files and 'simplex' not in files:
            # print(os.path.basename(os.path.normpath(files)))
            os.remove(files)

    return

def Recode_With_Binary_Labels(data_directory):

    file_database = glob.glob(os.path.join(data_directory, '*.nii*'))

    for filename in file_database:


        if 'autoAIF' not in filename and 'studyAIF' not in filename and 'same' not in filename and '_popAIF' not in filename:
            print(filename)
            split_file = str.split(filename, 'VISIT_0')
            new_filename = split_file[0] + 'VISIT_0' + split_file[1][0] + '_popAIF_' + split_file[1][2:]
            print(new_filename)
            # move(filename, new_filename)
            filename = new_filename


        if 't1map' not in filename and 't1static' not in filename:
            split_file = str.split(filename, 'VISIT_0')
            new_filename = split_file[0] + 'VISIT_0' + split_file[1][0] + '_t1static_' + split_file[1][2:]
            print(new_filename)
            move(filename, new_filename)
            filename = new_filename

        if 't1static_t1static_' in filename:
            new_filename = filename.replace('t1static_t1static_', 't1static_')
            print(filename)
            print(new_filename)
            move(filename, new_filename)
            filename = new_filename

        if 'sameAIF__autoAIF' in filename:
            new_filename = filename.replace('sameAIF__autoAIF', 'sameAutoAIF')
            print(filename)
            print(new_filename)
            # move(filename, new_filename)
            filename = new_filename

        if 'threshold_-1' in filename:
            new_filename = filename.replace('threshold_-1', 'threshold_none')
            move(filename, new_filename)
            print(new_filename)
            filename = new_filename

        if 'threshold_0.01' in filename:
            new_filename = filename.replace('threshold_0.01', 'threshold_PCA')
            move(filename, new_filename)
            print(new_filename)
            filename = new_filename

    return

def Create_Resource_Directories(CED_directory, NHX_directory, ROI_folder, AIF_folder, T1MAP_folder, DCE_Folder):

    """ Move all ROIs/AIFs/T1MAPs for the DCE script from their idiosyncratic locations
        to a set folder.
    """

    output_folders = [ROI_folder, AIF_folder, T1MAP_folder, DCE_Folder]

    for output_folder in output_folders:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    NHX_CED_dirs = glob.glob(os.path.join(CED_directory, 'CED*/')) + glob.glob(os.path.join(NHX_directory, 'NHX*/'))

    visits = ['VISIT_01', 'VISIT_02']

    for visit in visits:

        for subdir in NHX_CED_dirs:

            ROI = os.path.join(subdir, visit, 'ROISTATS', 'T1AxialPost', 'rT1AxialPostROI.nii')
            output_path = os.path.join(ROI_folder, os.path.basename(os.path.normpath(subdir)) + '_' + visit + '_ROI.nii')
            if os.path.exists(ROI):
                print(output_path)
                copy(ROI, output_path)

            # AIF = os.path.join(subdir, visit, 'MAPS', 'NORDIC_ICE_AIF.txt')
            # output_path = os.path.join(AIF_folder, os.path.basename(os.path.normpath(subdir)) + '_' + visit + '_AIF.txt')
            # if os.path.exists(AIF):
            #     print(output_path)
            #     copy(AIF, output_path)

            # T1MAP = os.path.join(subdir, visit, 'MAPS', 'T1inDCE.nii')
            # output_path = os.path.join(output_folder, os.path.basename(os.path.normpath(subdir)) + '_' + visit + '_T1inDCE.nii')
            # if os.path.exists(T1MAP):
            #     print(output_path)
            #     copy(T1MAP, output_path)

            DCE = os.path.join(subdir, visit, 'MAPS', 'dce_mc_st_eco1.nii')
            output_path = os.path.join(output_folder, os.path.basename(os.path.normpath(subdir)) + '_' + visit + '_DCE_ECHO1.nii')
            if os.path.exists(DCE):
                print(output_path)
                copy(DCE, output_path)

def Convert_NordicIce_AIF(AIF_directory, output_suffix='_AIF'):
    AIF_list = glob.glob(os.path.join(AIF_directory, '*VISIT*.txt'))
    AIF_numpy_list = [[np.loadtxt(AIF, dtype=float), AIF] for AIF in AIF_list]

    for AIF in AIF_numpy_list:
        print(AIF[1])
        print(AIF[0].shape)

        np.savetxt(replace_suffix(AIF[1], '', output_suffix), AIF[0][None], fmt='%2.5f', delimiter=';')


def Create_Study_AIF(AIF_directory, output_AIF):

    """ Average all AIFs into one AIF.
    """

    AIF_list = glob.glob(os.path.join(AIF_directory, '*VISIT*.txt'))

    AIF_numpy_list = [[np.loadtxt(AIF, delimiter=';', dtype=object), AIF] for AIF in AIF_list]

    AIF_array = np.zeros((len(AIF_numpy_list), 60), dtype=object)

    for row_idx, row in enumerate(AIF_array):
        print(len(AIF_numpy_list[row_idx][0]))
        print(AIF_numpy_list[row_idx][1])
        print(AIF_numpy_list[row_idx][0])
        AIF_array[row_idx, :] = AIF_numpy_list[row_idx][0][0:60]

    np.set_printoptions(suppress=True)

    AIF_array = AIF_array.astype(float)
    print(AIF_array.shape)
    print(np.mean(AIF_array, axis=0))
    print(np.mean(AIF_array, axis=0).T.shape)

    np.savetxt(output_AIF, np.mean(AIF_array, axis=0)[None], fmt='%2.5f', delimiter=';')

    return

def Create_Average_AIF(AIF_directory, output_AIF_directory):

    """ Create a patient-averaged AIF.
    """

    AIF_list = glob.glob(os.path.join(AIF_directory, '*VISIT*.txt'))

    for AIF_idx, AIF in enumerate(AIF_list):

        # print(AIF)

        if 'VISIT_01' in AIF:

            split_AIF = str.split(os.path.basename(AIF), '_')
            split_AIF[3] = '02'
            visit_2_AIF = os.path.join(AIF_directory, '_'.join(split_AIF))

            print(visit_2_AIF)

            if not os.path.exists(visit_2_AIF):
                continue

            AIF_numpy_1, AIF_numpy_2 = np.loadtxt(AIF, delimiter=';', dtype=object), np.loadtxt(visit_2_AIF, delimiter=';', dtype=object)

            print(AIF_numpy_1)
            print(AIF_numpy_2)

            output_AIF = (AIF_numpy_1[0:60].astype(float) + AIF_numpy_2[0:60].astype(float)) / 2.0
            output_filename = os.path.join(output_AIF_directory, '_'.join(split_AIF[0:4]) + '_AIF_average.txt')

            np.savetxt(output_filename, output_AIF[None], fmt='%2.5f', delimiter=';')

    return

def Store_Unneeded_Codes(data_directory, storage_directory):

    file_database = glob.glob(os.path.join(storage_directory, '*.nii*'))

    if not os.path.exists(storage_directory):
        os.mkdir(storage_directory)

    i = 0
    for file in file_database:

        if 'pca_0' in file and ('blur_0.2' in file or 'blur_0_' in file) and ('threshold_-1' in file or 'threshold_none' in file):
            # print(os.path.basename(file))
            move(file, os.path.join(data_directory, os.path.basename(file)))

    print(i)

def Store_and_Retrieve(data_directory, storage_directory):

    # Store
    file_database = glob.glob(os.path.join(data_directory, '*.nii*'))

    for file in file_database:

        if 'blur_0.8' in file or 'lm' in file or 'conv' in file:
            print(os.path.basename(file))
            move(file, os.path.join(storage_directory, os.path.basename(file)))

    # Retrieve
    file_database = glob.glob(os.path.join(storage_directory, '*.nii*'))

    for file in file_database:

        if 'pca_0' in file and ('blur_0.2' in file or 'blur_0_' in file) and ('simplex' in file and 'recursive' in file) and ('threshold_-1' in file or 'threshold_none' in file):
            print(os.path.basename(file))
            move(file, os.path.join(data_directory, os.path.basename(file)))

def Determine_R2_Cutoff_Point(input_directory, ROI_directory):

    """ Save ROI statistics into a giant csv file.
    """

    file_database = glob.glob(os.path.join(input_directory, '*.nii*'))

    output_headers = ['filename','mean','median','std','min','max','total_voxels','removed_values', 'removed_percent', 'low_values', 'low_percent']

    output_data = np.zeros((1+len(file_database), len(output_headers)),dtype=object)
    output_data[0,:] = output_headers

    ROI_dict = {}

    for ROI in glob.glob(os.path.join(ROI_directory, '*.nii*')):
        ROI_dict[os.path.basename(os.path.normpath(ROI))[0:15]] = convert_input_2_numpy(ROI)

    r2_masked_num, r2_total_num = [0]*100, [0]*100

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    for row_idx, filename in enumerate(file_database):

        if 'ktrans' not in filename or '0.2' in filename:
            continue

        data_array = convert_input_2_numpy(filename)
        r2_array = convert_input_2_numpy(replace_suffix(filename, input_suffix=None, output_suffix='r2', suffix_delimiter='_'))
        # print(replace_suffix(filename, input_suffix=None, output_suffix='r2', suffix_delimiter='_'))

        patient_visit_code = os.path.basename(os.path.normpath(filename))[0:15]
        roi_array = ROI_dict[patient_visit_code]

        for r2_idx, r2_threshold in enumerate(np.arange(0,1,.01)):
            r2_masked_num[r2_idx] += ((r2_array <= r2_threshold) & (roi_array > 0)).sum()
            r2_total_num[r2_idx] += (roi_array > 0).sum()

        print(np.array(r2_masked_num, dtype=float) / np.array(r2_total_num, dtype=float))

    r2_percent_num = np.array(r2_masked_num, dtype=float) / np.array(r2_total_num, dtype=float)

    for r2_idx, r2_threshold in enumerate(xrange(0,1,.01)):
        print(r2_threshold)
        print(r2_percent_num[r2_idx])

    return

def Rename_Files(input_directory):

    files = glob.glob(os.path.join(input_directory, '*.nii*'))
    for file in files:
        new_path = file.replace('0.2', '02')
        move(file, new_path)

def Preprocess_Volumes(input_directory, output_directory, r2_threshold=.9):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    file_database = glob.glob(os.path.join(input_directory, '*r2*.nii*'))
    print(os.path.join(input_directory, '*r2*.nii*'))

    for file in file_database:

        print(file)

        input_ktrans = replace_suffix(file, 'r2', 'ktrans')
        input_ve = replace_suffix(file, 'r2', 've')

        output_ktrans = os.path.join(output_directory, replace_suffix(os.path.basename(file), 'r2', 'ktrans_r2_' + str(r2_threshold)))
        output_ve = os.path.join(output_directory, replace_suffix(os.path.basename(file), 'r2', 've_r2_' + str(r2_threshold)))
        output_kep = os.path.join(output_directory, replace_suffix(os.path.basename(file), 'r2', 'kep_r2_' + str(r2_threshold)))
        output_r2 = os.path.join(output_directory, replace_suffix(os.path.basename(file), 'r2', 'r2_r2_' + str(r2_threshold)))

        print(input_ktrans)

        r2_map = np.nan_to_num(convert_input_2_numpy(file))
        ktrans_map = convert_input_2_numpy(input_ktrans)
        ve_map = convert_input_2_numpy(input_ve)

        print((r2_map < r2_threshold).sum())

        ve_map[ktrans_map > 10] = 0
        ktrans_map[ktrans_map > 10] = 0
        ktrans_map[ve_map > 1] = 0
        ve_map[ve_map > 1] = 0

        ktrans_map[r2_map < r2_threshold] = -.01
        ve_map[r2_map < r2_threshold] = -.01
        kep_map = np.nan_to_num(ktrans_map / ve_map)
        kep_map[r2_map < r2_threshold] = -.01

        save_numpy_2_nifti(ktrans_map, input_ktrans, output_ktrans)
        save_numpy_2_nifti(ve_map, input_ktrans, output_ve)
        save_numpy_2_nifti(kep_map, input_ktrans, output_kep)
        save_numpy_2_nifti(r2_map, input_ktrans, output_r2)


def Save_Directory_Statistics(input_directory, ROI_directory, output_csv, mask=False, mask_suffix='_mask', r2_thresholds=[.9]):

    """ Save ROI statistics into a giant csv file.
    """

    # exclude_patients = ['CED_19', ]

    file_database = glob.glob(os.path.join(input_directory, '*blur*r2_' + str(r2_thresholds[0]) + '.nii*'))

    output_headers = ['filename','mean','median','min','max','std', 'total_voxels','removed_values', 'removed_percent', 'low_values', 'low_percent']

    ROI_dict = {}

    for ROI in glob.glob(os.path.join(ROI_directory, '*.nii*')):
        ROI_dict[os.path.basename(os.path.normpath(ROI))[0:15]] = convert_input_2_numpy(ROI)

    for r2 in r2_thresholds:

        output_data = np.zeros((1+len(file_database), len(output_headers)),dtype=object)
        output_data[0,:] = output_headers

        with open(replace_suffix(output_csv, '', '_' + str(r2)), 'wb') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(output_data[0,:])

            for row_idx, filename in enumerate(file_database):

                data_array = convert_input_2_numpy(filename)
                patient_visit_code = os.path.basename(os.path.normpath(filename))[0:15]
                roi_array = ROI_dict[patient_visit_code]

                r2_filename = str.split(filename, '_')
                r2_filename[-3] = 'r2'
                r2_filename = '_'.join(r2_filename)

                r2_array = convert_input_2_numpy(r2_filename)

                data_array[data_array<0] = -.01
                data_array[r2_array<=r2] = -.01
                data_array[roi_array<=0] = -.01
                masked_data_array_ROI = np.ma.masked_where(data_array < 0, data_array)

                ROI_values = [np.ma.mean(masked_data_array_ROI), 
                                np.ma.median(masked_data_array_ROI), 
                                np.ma.min(masked_data_array_ROI), 
                                np.ma.max(masked_data_array_ROI), 
                                np.ma.std(masked_data_array_ROI),
                                (roi_array > 0).sum(), 
                                ((data_array <= 0) & (roi_array > 0)).sum(),
                                float(((data_array <= 0) & (roi_array > 0)).sum()) / float((roi_array > 0).sum()), 
                                ((r2_array >= r2) & (roi_array > 0)).sum(), 
                                float(((r2_array >= r2) & (roi_array > 0)).sum()) / float((roi_array > 0).sum())]

                print(ROI_values)

                output_data[row_idx+1] = [filename] + ROI_values

                csvfile.writerow(output_data[row_idx+1])

    return

def Paired_Visits_Worksheet(input_csv, output_csv, grab_column=2, r2_thresholds=[.9]):

    print(r2_thresholds)

    for r2 in r2_thresholds:

        input_data = np.genfromtxt(replace_suffix(input_csv, '', '_' + str(r2)), delimiter=',', dtype=object, skip_header=1)

        print(input_data)

        visit_1_list = [x for x in input_data[:,0] if 'VISIT_01' in x]

        output_data = np.zeros((len(visit_1_list)+1, 3), dtype=object)
        output_data[0,:] = ['method_code', 'visit_1', 'visit_2']

        with open(replace_suffix(output_csv, '', '_' + str(r2)), 'wb') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(output_data[0,:])

            for visit_idx, visit in enumerate(visit_1_list):

                if 'r2_r2' in visit:
                    continue

                split_visit = str.split(visit, 'VISIT_01')
                new_visit = split_visit[0] + 'VISIT_02' + split_visit[1]

                if new_visit in input_data[:,0]:
                    
                    print(np.where(input_data == visit)[0][0])

                    output_data[visit_idx+1, 0] = visit
                    output_data[visit_idx+1, 1] = input_data[np.where(input_data == visit)[0][0], grab_column]
                    output_data[visit_idx+1, 2] = input_data[np.where(input_data == new_visit)[0][0], grab_column]

                if output_data[visit_idx+1, 0] != 0 and output_data[visit_idx+1, 0] != '0' and input_data[np.where(input_data == visit)[0][0], -1] != '0' and input_data[np.where(input_data == new_visit)[0][0], -1] != '0':
                    csvfile.writerow(output_data[visit_idx+1,:])

def Coeffecient_of_Variation_Worksheet(input_csv, output_csv, r2_thresholds=[.9]):

    for r2 in r2_thresholds:

        input_data = np.genfromtxt(replace_suffix(input_csv, '', '_' + str(r2)), delimiter=',', dtype=object, skip_header=1)
        headers = ['method', 'RMS_COV', 'LOG_COV', 'SD_COV', 'CCC', 'R2', 'LOA_pos', 'LOS_neg', 'RC', 'mean_all_vals', 'n_measurements']
        output_data = np.zeros((3000, len(headers)), dtype=object)
        output_data[0,:] = headers
        methods, finished_methods = [], []

        # Get all methods
        for row in input_data:

            if row[0] == '0' or '--' in row:
                continue

            methods += [str.split(row[0], '/')[-1][15:]]

        method_dict = defaultdict(set)
        for method in methods:
            patient_list = [method == str.split(x, '/')[-1][15:] for x in input_data[:,0]]
            patient_list = input_data[patient_list, :]

            not_masked = [(x[1] != '--' and x[2] != '--') for x in patient_list]
            not_masked_patient_list = patient_list[not_masked, :]

            for row in not_masked_patient_list:
                method_dict[method].add(str.split(row[0], '/')[-1][0:15])

        available_patients = []
        for key, value in method_dict.iteritems():
            print(key)
            if len(value) < 5:
                continue
            if available_patients == []:
                available_patients = value
            if len(value) < len(available_patients):
                available_patients = value

        print(available_patients)
        print(len(available_patients))
        
        new_input_data = np.zeros((1,3), dtype=object)
        for row_idx, row in enumerate(input_data):
            patient = str.split(row[0], '/')[-1][0:15]
            if patient in available_patients:
                new_input_data = np.vstack((new_input_data, row))
        input_data = new_input_data[1:,:]

        with open(replace_suffix(output_csv, '', '_' + str(r2)), 'wb') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(output_data[0,:])

            row_idx = 0

            for row in input_data:

                if row[0] == '0' or '--' in row or row[0] == 0:
                    continue

                patient = str.split(row[0], '/')[-1][0:15]
                if patient not in available_patients:
                    continue

                method = str.split(row[0], '/')[-1][15:]

                if 't1map' in method:
                    continue

                aif_method = str.split(method, '_')
                aif_method[1] = 'sameAIF21'
                aif_method = '_'.join(aif_method)

                for row2 in input_data:
                    if aif_method in row2:
                        continue

                if method not in finished_methods:
                    # patient_list = np.where(method in input_data)
                    patient_list = [method == str.split(x, '/')[-1][15:] for x in input_data[:,0]]
                    patient_list = input_data[patient_list, :]
                    # print('METHOD', method)

                    # Non-Iterative Equations

                    not_masked = [(x[1] != 'nan' and x[2] != 'nan') for x in patient_list]
                    # print(not_masked)
                    not_masked_patient_list = patient_list[not_masked, :]
                    # print(not_masked_patient_list)
                    x, y = not_masked_patient_list[:,1].astype(float), not_masked_patient_list[:,2].astype(float)

                    if not_masked_patient_list.shape[0] < 10:
                        continue

                    # CCC
                    mean_x = np.mean(x)
                    mean_y = np.mean(y)
                    std_x = np.std(x)
                    std_y = np.std(y)
                    correl = np.ma.corrcoef(x,y)[0,1]
                    CCC = (2 * correl * std_x * std_y) / (np.ma.var(x) + np.ma.var(y) + np.square(mean_x - mean_y))

                    # Mean all values
                    mean_all_vals = np.mean(not_masked_patient_list[:,1:].astype(float))

                    # R2
                    R2_score = r2_score(y, x)

                    # Limits of Agreement (LOA)
                    differences = x - y
                    mean_diff = np.mean(differences)
                    std_diff = np.std(differences)
                    LOA_neg, LOA_pos = mean_diff - 2*std_diff, mean_diff + 2*std_diff

                    # Covariance and Repeatability Coeffecient
                    RMS_sum = 0
                    LOG_sum = 0
                    SD_sum_1 = 0
                    SD_sum_2 = 0
                    RC_sum = 0
                    n = 0

                    for patient in not_masked_patient_list:

                        data_points = [float(d) for d in patient[1:]]

                        skip=False
                        for d in data_points:
                            if d == 0:
                                skip = True
                        if skip:
                            continue

                        print(data_points)

                        RMS_sum += np.power(abs(data_points[0] - data_points[1]) / np.mean(data_points), 2)
                        LOG_sum += np.power(np.log(data_points[0]) - np.log(data_points[1]), 2)
                        SD_sum_1 += np.power(data_points[0] - data_points[1], 2)
                        SD_sum_2 += np.sum(data_points)
                        n += 1

                    RMS_COV = 100 * np.sqrt(RMS_sum / (2*n))
                    LOG_COV = 100 * np.exp(np.sqrt(LOG_sum / (2*n)) - 1)
                    SD_COV = 100 * np.sqrt(SD_sum_1 / (2*n)) / (SD_sum_2 / (2*n))
                    RC = (SD_sum_1 / n) * 1.96

                    output_data[row_idx+1, :] = [method, RMS_COV, LOG_COV, SD_COV, CCC, R2_score, LOA_pos, LOA_neg, RC, mean_all_vals, n]
                    # print(output_data[row_idx+1, :])

                    finished_methods += [method]
                    # print(methods)

                    if output_data[row_idx+1, 0] != 0 and output_data[row_idx+1, 0] != '0':
                        print('nice')
                        csvfile.writerow(output_data[row_idx+1,:])

                    row_idx += 1


                else:
                    # print('SKIPPED!!!!')
                    continue

    return

# def Equalize_Patient_Number(input_csv, output_csv, r2=.9):

#     input_data = np.genfromtxt(replace_suffix(input_csv, '', '_' + str(r2)), delimiter=',', dtype=object, skip_header=1)

#     output_data = np.zeros((len(visit_1_list)+1, 3), dtype=object)

#     for row in input_data:

#         patient_num = os.path.basename(row[0])
#         patient_num = patient_num[0:6]

#         if patient

#     return

if __name__ == '__main__':

    data_directory = '/home/abeers/Data/DCE_Package/Test_Results/OLD/New_AIFs/Echo1'
    storage_directory = '/home/abeers/Data/DCE_Package/Test_Results/Echo1/Storage'
    preprocess_directory = '/home/abeers/Data/DCE_Package/Test_Results/New_AIFs/Echo1/PreProcess'

    data_directory = '/home/abeers/Data/DCE_Package/Test_Results/Old_AIFs_Minor_Blur/Echo1'
    preprocess_directory = '/home/abeers/Data/DCE_Package/Test_Results/Old_AIFs_Minor_Blur/Echo1/Preprocess'

    NHX_directory = '/qtim2/users/data/NHX/ANALYSIS/DCE/'
    CED_directory = '/qtim/users/data/CED/ANALYSIS/DCE/PREPARATION_FILES/'
    
    ROI_directory = '/home/abeers/Data/DCE_Package/Test_Results/ROIs'
    AIF_directory = '/home/abeers/Data/DCE_Package/Test_Results/AIFs'
    T1MAP_directory = '/home/abeers/Data/DCE_Package/Test_Results/T1Maps'
    DCE_directory = '/home/abeers/Data/DCE_Package/Test_Results/DCE_Echo1'

    ALT_AIF_directory = '/home/abeers/Data/DCE_Package/Test_Results/DCE_Echo1/AIFS'

    output_csv = '/home/abeers/Requests/Jayashree/DCE_Repeatability_Data/DCE_Assay_Patient_Level_Statistics_old_blur.csv'
    paired_csv = '/home/abeers/Requests/Jayashree/DCE_Repeatability_Data/DCE_Assay_Visit_Level_Statistics_old_blur.csv'
    cov_csv = '/home/abeers/Requests/Jayashree/DCE_Repeatability_Data/DCE_Assay_Repeatability_Measures_old_blur.csv'

    r2_thresholds = [0.6]
    for r2 in r2_thresholds:

        Rename_Files(data_directory)
        Preprocess_Volumes(data_directory, preprocess_directory, r2_threshold = r2)
        # Determine_R2_Cutoff_Point(data_directory, ROI_directory)
        # Create_Average_AIF(ALT_AIF_directory, ALT_AIF_directory)
        # Rename_LM_Files(data_directory)
        # Copy_SameAIF_Visit1_Tumors(data_directory)
        # Convert_NordicIce_AIF(ALT_AIF_directory)
        # Create_Resource_Directories(CED_directory, NHX_directory, ROI_directory, AIF_directory, T1MAP_directory, DCE_directory)
        # Create_Study_AIF(ALT_AIF_directory, '/home/abeers/Data/DCE_Package/Test_Results/DCE_Echo1/AIFS/Study_AIF.txt')
        # Store_Unneeded_Codes(data_directory, storage_directory)
        Save_Directory_Statistics(preprocess_directory, ROI_directory, output_csv, r2_thresholds = [r2])
        # Reshape_Statisticts_Worksheet(output_csv, reshaped_output_csv, ROI_directory)
        # Delete_Extra_Files(data_directory)
        Paired_Visits_Worksheet(output_csv, paired_csv, r2_thresholds = [r2])
        Coeffecient_of_Variation_Worksheet(paired_csv, cov_csv, r2_thresholds = [r2])
        
        # Coeffecient_of_Variation_Worksheet(paired_reduced_csv, cov_reduced_csv)
        # Recode_With_Binary_Labels(data_directory)
        # Store_and_Retrieve(data_directory, storage_directory)

    pass
