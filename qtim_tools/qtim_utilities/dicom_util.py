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
from nifti_util import save_numpy_2_nifti, check_image_2d
import matplotlib.pyplot as plt

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
    
def dcm_2_nifti(input_folder, output_folder, verbose=True, naming_tags=['SeriesDescription'], prefix='', suffix='_harden', write_header=False, header_suffix='_header', harden_orientation=True):

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

        np.set_printoptions(suppress=True)

        # Grab DICOMs for a certain Instance
        current_dicoms = unique_dicoms[UID]

        # Sort DICOMs by Instance.
        dicom_instances = [x.data_element('InstanceNumber').value for x in current_dicoms]
        current_dicoms = [x for _,x in sorted(zip(dicom_instances,current_dicoms))]
        first_dicom, last_dicom = current_dicoms[0], current_dicoms[-1]

        # Create a filename for the DICOM
        volume_label = '_'.join([first_dicom.data_element(tag).value for tag in naming_tags]).replace(" ", "")
        volume_label = prefix + "".join([c for c in volume_label if c.isalpha() or c.isdigit() or c==' ']).rstrip() + suffix + '.nii.gz'

        if verbose:
            print 'Saving...', volume_label

        try:
            # Extract patient position information for affine creation.
            output_affine = np.eye(4)
            image_position_patient = np.array(first_dicom.data_element('ImagePositionPatient').value).astype(float)
            image_orientation_patient = np.array(first_dicom.data_element('ImageOrientationPatient').value).astype(float)
            last_image_position_patient = np.array(last_dicom.data_element('ImagePositionPatient').value).astype(float)
            pixel_spacing_patient = np.array(first_dicom.data_element('PixelSpacing').value).astype(float)

            # Create DICOM Space affine (don't fully understand, TODO)
            output_affine[0:3, 0] = pixel_spacing_patient[0] * image_orientation_patient[0:3]
            output_affine[0:3, 1] = pixel_spacing_patient[1] * image_orientation_patient[3:6]
            output_affine[0:3, 2] = (image_position_patient - last_image_position_patient) / (1 - len(current_dicoms))
            output_affine[0:3, 3] = image_position_patient

            # Transformations from DICOM to Nifti Space (don't fully understand, TOO)
            cr_flip = np.eye(4)
            cr_flip[0:2,0:2] = [[0,1],[1,0]]
            neg_flip = np.eye(4)
            neg_flip[0:2,0:2] = [[-1,0],[0,-1]]
            output_affine = np.matmul(neg_flip, np.matmul(output_affine, cr_flip))

            # Create numpy array data...
            output_numpy = np.zeros((current_dicoms[0].pixel_array.shape + (len(current_dicoms),)), dtype=float)
            for i in xrange(output_numpy.shape[-1]):
                output_numpy[..., i] = current_dicoms[i].pixel_array

            # If preferred, harden to identity matrix space (LPS, maybe?)
            # Also unsure of the dynamic here, but they work.
            if harden_orientation is not None:

                cx, cy, cz = np.argmax(np.abs(output_affine[0:3,0:3]), axis=0)
                rx, ry, rz = np.argmax(np.abs(output_affine[0:3,0:3]), axis=1)

                output_numpy = np.transpose(output_numpy, (rx,ry,rz))

                harden_matrix = np.eye(4)
                for dim, i in enumerate([cx,cy,cz]):
                    harden_matrix[i,i] = 0
                    harden_matrix[dim, i] = 1
                output_affine = np.matmul(output_affine, harden_matrix)

                flip_matrix = np.eye(4)
                for i in xrange(3):
                    if output_affine[i,i] < 0:
                        flip_matrix[i,i] = -1
                        output_numpy = np.flip(output_numpy, i)

                output_affine = np.matmul(output_affine, flip_matrix)

            save_numpy_2_nifti(output_numpy, output_affine, os.path.join(output_folder, volume_label))

        except:
            print 'Could not read DICOM at SeriesDescription...', volume_label

if __name__ == '__main__':
    pass