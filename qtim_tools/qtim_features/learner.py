""" This 'learner' program, perhaps wrongly named, is the my current
    general utility function. It takes in a folder full of images and
    labels, parses them into numpy arrays, extracts features from those
    arrays, and writes them into an easily accessible .csv. As of yet,
    it does not do any learning..
"""

# import 

import GLCM
import morphology
import statistics


from qtim_tools.qtim_utilities import nifti_util

import glob
import os
import numpy as np
import nibabel as nib
import csv
import fnmatch
from shutil import copy, move

feature_dictionary = {'GLCM': GLCM, 'morphology': morphology, 'statistics': statistics}

def generate_feature_list_batch(folder, features=['GLCM', 'morphology', 'statistics'], recursive=False, labels=False, label_suffix="-label", decisions=False, levels=255, mask_value=0, use_labels=[-1], erode=0, filenames=True, featurenames=True, outfile='', overwrite=True, clear_file=True, write_empty=True, test=False):

    total_features, feature_indexes, label_output = generate_feature_indices(features, featurenames)

    # This needs to be restructured, probably with a new method to iterate through images. Currently, this will not work
    # wtihout an output file. The conflict is between retaining the ability to append to files in real-time (to prevent
    # catastrophic errors from wasting eons of processing time) and having a conditional "outfile" parameter.

    if outfile != '':
        outfile = determine_outfile_name(outfile, overwrite)

        if clear_file:
            open(outfile, 'w').close()

        with open(outfile, 'ab') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(label_output[0,:])

            imagepaths, label_images = generate_filename_list(folder, labels, label_suffix, recursive)
            
            numerical_output = np.zeros((1, total_features), dtype=float)
            index_output = np.zeros((1, 1), dtype=object)

            for imagepath in imagepaths:

                print '\n'
                print 'Pre-processing data...'

                image_list, unmodified_image_list, imagename_list, attributes_list = generate_numpy_images(imagepath, labels=labels, label_suffix=label_suffix, label_images=label_images, levels=levels, mask_value=mask_value, use_labels=use_labels, erode=erode)
                
                if image_list == []:
                    if write_empty:
                        empty_output = np.zeros((1, total_features + 1), dtype=object)
                        empty_output[0,0] = imagepath
                        csvfile.writerow(empty_output[0,:])
                        continue

                print 'Pre-processing complete!'

                for image_idx, image in enumerate(image_list):

                    print ''
                    print 'Working on image...'
                    print imagename_list[image_idx]
                    print 'Voxel sum...'
                    print np.sum(image)
                    print 'Image shape...'
                    print image.shape

                    if filenames:
                        index = imagename_list[image_idx]
                    else:
                        index = numerical_output.shape[0]

                    if numerical_output[0,0] == 0:
                        numerical_output[0, :] = generate_feature_list_method(image, unmodified_image_list[image_idx], attributes_list[image_idx], features, feature_indexes, total_features, levels, mask_value=0)
                        index_output[0,:] = index
                    else:
                        numerical_output = np.vstack((numerical_output, generate_feature_list_method(image, unmodified_image_list[image_idx], attributes_list[image_idx], features, feature_indexes, total_features, levels, mask_value=0)))
                        index_output = np.vstack((index_output, index))

                    csvfile.writerow(np.hstack((index_output[-1,:], numerical_output[-1,:])))

    final_output = np.hstack((index_output, numerical_output))

    print 'Feature writing complete, writing output...'
    print '\n'

    print final_output

    return final_output

def generate_feature_list_single(vol_filename, features=['GLCM', 'morphology', 'statistics'], labels=False, label_filename='',label_suffix="-label", decisions=False, levels=255, filenames=True, featurenames=True, outfile='', overwrite=True, write_empty=True, mask_value=0, test=False, use_labels=[-1], erode=0):
    
    total_features, feature_indexes, label_output = generate_feature_indices(features, featurenames)

    if outfile != '':
        outfile = determine_outfile_name(outfile, overwrite)

        with open(outfile, 'ab') as writefile:
            csvfile = csv.writer(writefile, delimiter=',')
            csvfile.writerow(label_output[0,:])
            
            numerical_output = np.zeros((1, total_features), dtype=float)
            index_output = np.zeros((1, 1), dtype=object)

            final_output = write_image_method(vol_filename, label_filename, csvfile, total_features, features, feature_indexes, numerical_output, index_output, labels=False, label_suffix='-label', levels=100, mask_value=0, use_labels=[-1], erode=0, write_empty=False)

    print 'Feature writing complete, writing output...'
    print '\n'

    print final_output

    return final_output

def write_image_method(imagepath, label_images, csvfile, total_features, features, feature_indexes, numerical_output, index_output, labels=False, label_suffix='-label', levels=100, mask_value=0, use_labels=[-1], erode=0, write_empty=False):

    # This function is a bit clumsy. So many parameters..

    print '\n'
    print 'Pre-processing data...'

    image_list, unmodified_image_list, imagename_list, attributes_list = generate_numpy_images(imagepath, labels=labels, label_suffix=label_suffix, label_images=label_images, levels=levels, mask_value=mask_value, use_labels=use_labels, erode=erode)
    
    if image_list == []:
        if write_empty:
            empty_output = np.zeros((1, total_features + 1), dtype=object)
            empty_output[0,0] = imagepath
            print 'Writing empty row in place of missing data...'
            csvfile.writerow(empty_output[0,:])

    else:

        print 'Pre-processing complete!'

        for image_idx, image in enumerate(image_list):

            print ''
            print 'Working on image...'
            print imagename_list[image_idx]
            print 'Voxel sum...'
            print np.sum(image)
            print 'Image shape...'
            print image.shape

            if filenames:
                index = imagename_list[image_idx]
            else:
                index = numerical_output.shape[0]

            if numerical_output[0,0] == 0:
                numerical_output[0, :] = generate_feature_list_method(image, unmodified_image_list[image_idx], attributes_list[image_idx], features, feature_indexes, total_features, levels, mask_value=0)
                index_output[0,:] = index
            else:
                numerical_output = np.vstack((numerical_output, generate_feature_list_method(image, unmodified_image_list[image_idx], attributes_list[image_idx], features, feature_indexes, total_features, levels, mask_value=0)))
                index_output = np.vstack((index_output, index))

            csvfile.writerow(np.hstack((index_output[-1,:], numerical_output[-1,:])))
    
    return np.hstack((index_output, numerical_output))

def determine_outfile_name(outfile, overwrite=True):
    
    write_flag = False
    while not write_flag:
        if not os.path.isfile(outfile):
            write_flag = True
            continue
        if overwrite:
            write_flag = True
        else:
            split_outfile = str.split(outfile,'.')
            print split_outfile
            outfile = '.'.join(split_outfile[0:-1]) + '_new.' + split_outfile[-1]
            if not os.path.isfile(outfile):
                write_flag = True

    return outfile

def generate_feature_indices(features=['GLCM', 'morphology', 'statistics'], featurenames=True):

    total_features = 0
    feature_indexes = [0]

    for feature in features:
        total_features += feature_dictionary[feature].feature_count()
        if feature_indexes == [0]:
            feature_indexes = [0, feature_dictionary[feature].feature_count()]
        else:
            feature_indexes += [feature_indexes[-1] + feature_dictionary[feature].feature_count()]
    
    if featurenames:
        label_output = np.zeros((1, total_features+1), dtype=object)
        for feature_idx, feature in enumerate(features):
            label_output[0, (1+feature_indexes[feature_idx]):(1+feature_indexes[feature_idx+1])] = feature_dictionary[feature].featurename_strings()
        label_output[0,0] = 'index'

    return [total_features, feature_indexes, label_output]

def generate_filename_list(folder, labels=False, label_suffix='-label', recursive=False):

    if recursive:
        imagepaths = []
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, '*.nii*'):
                imagepaths.append(os.path.join(root, filename))
    else:
        imagepaths = glob.glob(os.path.join(folder, "*.nii*"))

    # A bit redundant; this step and the previous step could probably be combined.
    imagepaths = [x for x in imagepaths if (x.endswith('.nii') or x.endswith('.nii.gz'))]

    if labels:
        label_images = [ x for x in imagepaths if label_suffix in x ]
        print os.path.join(folder, "*" + label_suffix + ".nii*")

    imagepaths = [ x for x in imagepaths if label_suffix not in x ]

    if imagepaths == []:
        raise ValueError("There are no .nii or .nii.gz images in the provided folder.")
    if labels and label_images == []:
        raise ValueError("There are no labels with the provided suffix in this folder. If you do not want to use labels, set the \'labels\' flag to \'False\'. If you want to change the label file suffix (default: \'-label\'), then change the \'label_suffix\' flag.")

    return [imagepaths, label_images]

def generate_numpy_images(imagepath, labels=False, label_suffix='-label', label_images=[], mask_value=0, levels=255, use_labels=[-1], erode=0):

    print imagepath
    print labels
    print label_suffix
    print label_images
    print mask_value
    print levels
    print use_labels
    print erode

    image_list = []
    unmodified_image_list = []
    imagename_list = []
    attributes_list = []
    
    # nifti_util.save_alternate_nifti(imagepath, levels, mask_value=mask_value)
    image = nifti_util.nifti_2_numpy(imagepath)

    # This is likely redundant with the basic assert function in nifti_util
    if not nifti_util.assert_3D(image):
        print 'Warning: image at path ' + imagepath + ' has multiple time points or otherwise greater than 3 dimensions, and will be skipped.'
        return [[],[],[],[]]

    if labels:

        if label_suffix == '':
            label_path = label_images
        else:
            split_path = str.split(imagepath, '.')
            label_path = split_path[0] + label_suffix + '.' + '.'.join(split_path[1:])

        if os.path.isfile(label_path):
            label_image = nifti_util.nifti_2_numpy(label_path)

            if label_image.shape != image.shape:
                print 'Warning: image and label do not have the same dimensions. Imaging padding support has not yet been added. This image will be skipped.'
                return [[],[],[],[]]

            # In the future: create an option to analyze each frame separately.
            if not nifti_util.assert_3D(label_image):
                print 'Warning: image at path ' + imagepath + ' has multiple time points or otherwise greater than 3 dimensions, and will be skipped.'
                return [[],[],[],[]]

            label_image = label_image.astype(int)
            label_indices = np.unique(label_image)

            if label_indices.size == 1:
                print 'Warning: image at path ' + imagepath + ' has an empty label-map, and will be skipped.'
                return[[],[],[],[]]

            # Will break if someone puts in '0' as a label to use.
            if use_labels[0] != -1:
                label_indices = np.array([0] + [x for x in label_indices if x in use_labels])

            masked_images = nifti_util.mask_nifti(image, label_image, label_indices, mask_value=mask_value)

            for masked_image in masked_images:

                # nifti_util.check_tumor_histogram(masked_image, second_image_numpy=image, mask_value=mask_value, image_name = str.split(imagepath, '\\')[-1])
                # nifti_util.check_image(masked_image, mode="maximal_slice")

                masked_image = nifti_util.coerce_levels(masked_image, levels=levels, reference_image=image, method="divide", mask_value=mask_value)

                # nifti_util.check_image(masked_image, mode="maximal_slice")

                unmodified_image_list += [np.copy(masked_image)]

                # It would be nice in the future to check if an image is too small to erode. Maybe a minimum-size parameter?
                # Or maybe a "maximum volume reduction by erosion?" Hmm..
                masked_image = nifti_util.erode_label(masked_image, iterations=erode)

                # nifti_util.check_image(masked_image, mode="maximal_slice")

                image_list += [masked_image]

            filename = str.split(label_path, '\\')[-1]

            if label_indices.size == 2:
                imagename_list += [filename]
            else:
                split_filename = str.split(filename, '.')
                for labelval in label_indices[1:]:
                    filename = split_filename[0] + '_' + str(int(labelval)) + '.' + split_filename[1]
                    imagename_list += [filename]

            attributes_list += [nifti_util.return_nifti_attributes(imagepath)] * (label_indices.size - 1)
            print 'Finished... ' + str.split(imagepath, '\\')[-1]

        else:
            print 'Warning: image at path ' + imagepath + ' has no label-map, and will be skipped.'
            return[[],[],[],[]]

    else:
        image = nifti_util.coerce_levels(image, levels=levels, reference_image='', method="divide", mask_value=mask_value)
        image_list += [image]
        unmodified_image_list += [image]
        imagename_list += [imagepath]
        attributes_list += [nifti_util.return_nifti_attributes(imagepath)]

    return [image_list, unmodified_image_list, imagename_list, attributes_list]

def generate_feature_list_method(image, unmodified_image, attributes, features, feature_indexes='', total_features='', levels=-1, mask_value=0):

    if feature_indexes == '' or total_features == '':
        total_features = 0
        feature_indexes = [0]

        for feature in features:
            total_features += feature_dictionary[feature].feature_count()
            if feature_indexes == [0]:
                feature_indexes = [0, feature_dictionary[feature].feature_count()]
            else:
                feature_indexes += [feature_indexes[-1] + feature_dictionary[feature].feature_count()]

    numerical_output = np.zeros((1, total_features), dtype=float)

    if (image != mask_value).sum() == 0:
        print 'Warning: image is empty, either because it could not survive erosion or because of another error. It will be skipped.'
        return numerical_output

    for feature_idx, feature in enumerate(features):

        if feature == 'GLCM':

            # nifti_util.check_tumor_histogram(image, mask_value)
            # nifti_util.check_image(image, mode="maximal_slice")

            glcm_image = np.copy(image)
            glcm_image = glcm_image.astype(int)
            levels += 1
            print 'Calculating GLCM...'
            numerical_output[0, feature_indexes[feature_idx]:feature_indexes[feature_idx+1]] = GLCM.glcm_features(glcm_image, levels=levels)

        if feature == 'morphology':

            print 'Calculating morphology features...'
            numerical_output[0, feature_indexes[feature_idx]:feature_indexes[feature_idx+1]] = morphology.morphology_features(unmodified_image, attributes)

        if feature == 'statistics':

            print 'Calculating statistical features...'
            numerical_output[0, feature_indexes[feature_idx]:feature_indexes[feature_idx+1]] = statistics.statistics_features(image)

    print '\n'

    return numerical_output

def test_method():
    generate_feature_list_batch(folder='C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/test_data/test_data_features', labels=True, levels=100, outfile='test_feature_results.csv',test=False, mask_value=0, erode=[3,3,0], overwrite=True)
    return

def Elizabeth_Mets_ROIS():
    generate_feature_list_batch(folder='C:/Users/azb22/Documents/Scripting/Elizabeth_Mets_Heterogeneity/Melanoma_Mets_with_ROIs', labels=True, levels=100, outfile='mets_feature_results.csv',test=False, mask_value=0, erode=[0,0,0], overwrite=True, recursive=True, label_suffix=".nii-label", write_empty=False)
    return

if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=2)
    Elizabeth_Mets_ROIS()
    # test_method()
    # generate_feature_list_batch(folder='', labels=True, levels=100, outfile='',test=False, mask_value=0, erode=[3,3,0])