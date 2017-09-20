""" This is a utility module for loading dicom data and headers. It borrows heavily from pydicom.
    It will likely take a lot of DICOM knowledge to navigate..
"""

import dicom
import nibabel as nib
import numpy as np
import os
import glob
import re
from file_util import human_sort, grab_files_recursive
from collections import defaultdict
from nifti_util import save_numpy_2_nifti

def get_dicom_dictionary(input_filepath=[], dictionary_regex="*", return_type='name'):

    """ Returns a dictionary of dicom tags using pydicom. TODO: let users return tag
        dictionary.
    """

    if os.path.isdir(input_filepath):
        dictionary_file = glob.glob(os.path.join(input_filepath, '*'))[0]
    else:
        dictionary_file = input_filepath

    img_dicom = dicom.read_file(dictionary_file)

    output_dictionary = {}

    for key in img_dicom.dir():
        try:
            if key != 'PixelData':
                output_dictionary[key] = img_dicom.data_element(key).value
        except:
            pass

    return output_dictionary

def dcm_2_numpy(folder, return_header=False, verbose=True):

    """ Uses pydicom to stack an alphabetical list of DICOM files. TODO: Make it
        take slice_order into account.
    """

    if verbose:
        print 'Searching for dicom files...'

    found_files = grab_files_recursive(folder)

    if verbose:
        print 'Found', len(found_files), 'in directory. \n'
        print 'Checking DICOM compatability...'

    dicom_files = []
    for file in found_files:
        try:
            dicom_files += [dicom.read_file(file)]
        except:
            pass

    if verbose:
        print 'Found', len(dicom_files), 'DICOM files in directory. \n'
        print 'Counting volumes..'

    dicom_headers = [] 
    unique_dicoms = defaultdict(list)
    for dicom_file in dicom_files:
        UID = dicom_file.data_element('SeriesInstanceUID').value
        unique_dicoms[UID] += [dicom_file]

    if verbose:
        print 'Found', len(unique_dicoms.keys()), 'unique volumes \n'
        print 'Saving out files from these volumes.'

    output_dict = {}

    for UID in unique_dicoms.keys():
        
        current_dicoms = unique_dicoms[UID]
        volume_label = current_dicoms[0].data_element('SeriesDescription').value

        try:
            output_numpy = np.zeros((current_dicoms[0].pixel_array.shape + (len(current_dicoms),)), dtype=float)
            output_dict[volume_label] = output_numpy

            dicom_instances = [x.data_element('InstanceNumber').value for x in current_dicoms]
            current_dicoms = [x for _,x in sorted(zip(dicom_instances,current_dicoms))]

            for i in xrange(output_numpy.shape[-1]):
                output_numpy[..., i] = current_dicoms[i].pixel_array
        except:
            output_dict[volume_label] = None
            print 'Could not read DICOM at SeriesDescription...', volume_label

    return output_dict
    
def dcm_2_nifti(input_folder, output_folder, verbose=True, naming_tags=['SeriesDescription'], prefix='', suffix='', write_header=False, header_suffix='_header'):

    """ Uses pydicom to stack an alphabetical list of DICOM files. TODO: Make it
        take slice_order into account.
    """

    if verbose:
        print 'Searching for dicom files...'

    found_files = grab_files_recursive(input_folder)

    if verbose:
        print 'Found', len(found_files), 'in directory. \n'
        print 'Checking DICOM compatability...'

    dicom_files = []
    for file in found_files:
        try:
            dicom_files += [dicom.read_file(file)]
        except:
            continue

    if verbose:
        print 'Found', len(dicom_files), 'DICOM files in directory. \n'
        print 'Counting volumes..'

    dicom_headers = [] 
    unique_dicoms = defaultdict(list)
    for dicom_file in dicom_files:
        UID = dicom_file.data_element('SeriesInstanceUID').value
        unique_dicoms[UID] += [dicom_file]

    if verbose:
        print 'Found', len(unique_dicoms.keys()), 'unique volumes \n'
        print 'Saving out files from these volumes.'

    for UID in unique_dicoms.keys():


        current_dicoms = unique_dicoms[UID]

        # Sort DICOMs by Instance.
        dicom_instances = [x.data_element('InstanceNumber').value for x in current_dicoms]
        current_dicoms = [x for _,x in sorted(zip(dicom_instances,current_dicoms))]
        first_dicom, last_dicom = current_dicoms[0], current_dicoms[-1]

        volume_label = '_'.join([first_dicom.data_element(tag).value for tag in naming_tags]).replace(" ", "")
        volume_label = prefix + "".join([c for c in volume_label if c.isalpha() or c.isdigit() or c==' ']).rstrip() + suffix + '.nii.gz'

        if verbose:
            print 'Saving...', volume_label

        # try:
        # Create affine...
        output_affine = np.eye(4)
        image_position_patient = np.array(first_dicom.data_element('ImagePositionPatient').value).astype(float)
        image_orientation_patient = np.array(first_dicom.data_element('ImageOrientationPatient').value).astype(float)
        last_image_position_patient = np.array(last_dicom.data_element('ImagePositionPatient').value).astype(float)
        pixel_spacing_patient = np.array(first_dicom.data_element('PixelSpacing').value).astype(float)

        # image_orientation_patient = np.multiply(image_orientation_patient, [-1,-1,1,-1])

        print image_position_patient, last_image_position_patient
        print image_orientation_patient

        output_affine[0:3, 0] = pixel_spacing_patient[0] * image_orientation_patient[0:3]
        output_affine[0:3, 1] = pixel_spacing_patient[1] * image_orientation_patient[3:6]
        output_affine[0:3, 2] = (image_position_patient - last_image_position_patient) / (1 - len(current_dicoms))
        output_affine[0:3, 3] = image_position_patient

        x, y, z = np.argmax(np.abs(output_affine[0:3,0:3]), axis=0)

        print x,y,z
        cr_flip = np.eye(4)
        cr_flip[0:2,0:2] = [[0,1],[1,0]]
        neg_flip = np.eye(4)
        neg_flip[0:2,0:2] = [[-1,0],[0,-1]]
        # cr_flip[x, y] = 1
        # cr_flip[y, x] = 1 
        # cr_flip[x, x] = 0
        # cr_flip[y, y] = 0

        output_affine = np.matmul(neg_flip, np.matmul(output_affine, cr_flip))
        # output_affine[0:2,0:2] = output_affine[0:2,0:2]*-1
        print output_affine

        # print nib.orientations.io_orientation(output_affine)

        # Create array...
        output_numpy = np.zeros((current_dicoms[0].pixel_array.shape + (len(current_dicoms),)), dtype=float)
        for i in xrange(output_numpy.shape[-1]):
            output_numpy[..., i] = current_dicoms[i].pixel_array

        output_nifti = nib.Nifti1Image(output_numpy, output_affine)

        save_numpy_2_nifti(output_numpy, output_affine, os.path.join(output_folder, volume_label))

        # except:
            # print 'Could not read DICOM at SeriesDescription...', volume_label

    # for dicom_idx, dicom_file in enumerate(dicom_files):
    #     output_numpy[..., dicom_idx] = dicom.read_file(dicom_file).pixel_array

    # return output_numpy

if __name__ == '__main__':
    pass

    # image_numpy = nib.load('C:/Users/azb22/Documents/Scripting/Breast_MRI_Challenge/ISPY_data/AllFiles/ISPY1_1098_19860919/ISPY1_1098_19860919_BreastTissue.nii.gz')
    # transform_matrix = image_numpy.affine

    # if folder != []:
    #     img_dicom_list = []
    #     for root, dirnames, filenames in os.walk(folder):
    #         for filename in fnmatch.filter(filenames, attributes_regex):
    #             img_dicom_list.append(os.path.join(root, filename))

    #     if img_dicom_list == []:
    #         print "No DICOM attributes returned. Folder is empty."
    #     else:

    #         attribute_list = np.zeros((1, 10), dtype=object)
    #         temp_list = np.zeros((1, 10), dtype=object)
    #         attribute_list[0,:] = ['filename','PatientName','date','center1', 'center2', 'center3', 'lx','ly','lz','type']
    #         RadiusMatrix = np.zeros((3,3), dtype=float)

    #         for filepath_idx, filepath in enumerate(img_dicom_list):
    #             img_dicom = dicom.read_file(filepath)
    #             try:
    #                 # print img_dicom[0x117,0x1020][0][0x117,0x1043].value
    #                 RadiusMatrix[0,:] = img_dicom[0x117,0x1020][0][0x117,0x1043].value[0:3]
    #                 RadiusMatrix[1,:] = img_dicom[0x117,0x1020][0][0x117,0x1044].value[0:3]
    #                 RadiusMatrix[2,:] = img_dicom[0x117,0x1020][0][0x117,0x1045].value[0:3]
    #                 # print filepath
    #                 temp_list[0, 3] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[0]
    #                 temp_list[0, 4] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[1]
    #                 temp_list[0, 5] = img_dicom[0x117,0x1020][0][0x117,0x1042].value[2]
    #                 temp_list[0, 6] = np.max(abs(RadiusMatrix[:,0]))
    #                 temp_list[0, 7] = np.max(abs(RadiusMatrix[:,1]))
    #                 temp_list[0, 8] = np.max(abs(RadiusMatrix[:,2]))
    #                 # print filepath
    #                 temp_list[0, 9] = img_dicom[0x117,0x1020][0][0x117,0x1046].value
    #                 temp_list[0, 2] = img_dicom.StudyDate
    #                 temp_list[0, 1] = img_dicom.PatientID
    #                 temp_list[0, 0] = filepath
    #                 if [img_dicom.StudyDate, img_dicom.PatientID] not in attribute_list[:,1:3]:
    #                     attribute_list = np.vstack((attribute_list, temp_list))
    #                     # print temp_list
    #                 # print ''
    #             except:
    #                 continue
    #             if attribute_list[0, 7] == 'nan':
    #                 # continue
    #                 for i in xrange(attribute_list.shape[1]):
    #                     print attribute_list[0, i]
    #                 for i in xrange(2,6):
    #                     col_vec = attribute_list[0, i][0:3]
    #                     # col_vec[0]  = -col_vec[0]
    #                     # col_vec[1] = -col_vec[1]
    #                     # print np.round(nib.affines.apply_affine(transform_matrix, col_vec))
    #                 # print col_vec.shape
    #                 # for i in xrange(1,5):
    #                     # transformed = nib.affines.apply_affine(np.linalg.inv(transform_matrix), np.reshape(attribute_list[0, i][0:3], (1,3)))
    #                     # print transformed
    #                     # image = image_numpy.get_data()

    #             # fig = plt.figure()
    #             # imgplot = plt.imshow(image[:,:,int(transformed[0][2])], interpolation='none', aspect='auto')
    #             # plt.show()
                
    #             # print transform_matrix
    #             # print attribute_list[0, 1]
    #         with open('C:/Users/azb22/Documents/GitHub/QTIM_Pipelines/QTIM_Feature_Extraction_Pipeline/Test_Data/VOI_INFO_ISPY1.csv', 'wb') as writefile:
    #             csvfile = csv.writer(writefile, delimiter=',')
    #             for row in attribute_list:
    #                 csvfile.writerow(row)

    #     # print attribute_list

    # elif filepath != []:
    #     img_dicom = dicom.read_file(filepath) 
    #     print img_dicom

    # else:
    #     print "No DICOM attributes returned. Please provide a file or folder path."
    #     return