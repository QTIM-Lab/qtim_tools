import os
import glob
import numpy as np
import subprocess
import csv

from optparse import OptionParser
from subprocess import call
from shutil import copytree, rmtree
import nibabel as nib

def Compare_Segmentations(InputSeg, InputGroundTruth, InputVolume, OutputFolder, OutputSheet):

    # Create Folder variables.
    InputSeg_Base = os.path.basename(os.path.normpath(os.path.abspath(InputSeg))).replace(" ", "")
    InputGroundTruth_Base = os.path.basename(os.path.normpath(os.path.abspath(InputGroundTruth))).replace(" ", "")
    InputVolume_Base = os.path.basename(os.path.normpath(os.path.abspath(InputVolume))).replace(" ", "")
    mount_path = os.path.abspath(OutputFolder)

    # Create required mount folder for Plastimatch Docker
    if os.path.isdir(mount_path):
        rmtree(mount_path)
    os.mkdir(mount_path)

    # Convert CT
    copytree(InputVolume, os.path.join(mount_path,InputVolume_Base))
    Docker_Command = 'docker run --rm -v ' + mount_path + ':/input_dicom mayoqin/plastimatch plastimatch convert --input /input_dicom/' + InputVolume_Base + ' --output-img ./input_dicom/' + str.split(InputVolume_Base, '.')[0] + '.nii.gz'
    call(Docker_Command, shell=True)

    # Convert Segmentation
    copytree(InputSeg, os.path.join(mount_path,InputSeg_Base))
    Docker_Command = 'docker run --rm -v ' + mount_path + ':/input_dicom mayoqin/plastimatch plastimatch convert --input /input_dicom/' + InputSeg_Base + ' --prefix-format nii.gz --output-prefix ./input_dicom/' + str.split(InputSeg_Base, '.')[0] + '_split --fixed ./input_dicom/' + str.split(InputVolume_Base, '.')[0] + '.nii.gz --output-ss-list ./input_dicom/' + str.split(InputSeg_Base, '.')[0] + '.txt'
    call(Docker_Command, shell=True)

    # Convert Groundtruth
    copytree(InputGroundTruth, os.path.join(mount_path,InputGroundTruth_Base))
    Docker_Command = 'docker run --rm -v ' + mount_path + ':/input_dicom mayoqin/plastimatch plastimatch convert --input /input_dicom/' + InputGroundTruth_Base + ' --prefix-format nii.gz --output-prefix ./input_dicom/' + str.split(InputGroundTruth_Base, '.')[0] + '_split --fixed ./input_dicom/' + str.split(InputVolume_Base, '.')[0] + '.nii.gz --output-ss-list ./input_dicom/' + str.split(InputGroundTruth_Base, '.')[0] + '.txt'
    call(Docker_Command, shell=True)

    # Start by recording all of the labels available in Ground Truth Data
    Labels = {}
    for label_idx, Ground_Truth_Label in enumerate(glob.glob(os.path.join(mount_path, str.split(InputGroundTruth_Base, '.')[0] + '_split', '*'))):
        Label_Name = str.split(os.path.basename(os.path.normpath(Ground_Truth_Label)), '.')[0]
        Labels[Label_Name] = label_idx

    # Crop spinal cord by esophagus
    reference_esophagus = os.path.join(mount_path, str.split(InputGroundTruth_Base, '.')[0] + '_split', 'Esophagus.nii.gz')
    if os.path.exists(reference_esophagus):
        reference_esophagus_nifti = nib.load(reference_esophagus)
        reference_esophagus_numpy = reference_esophagus_nifti.get_fdat)
        reference_axial_dim = reference_esophagus_nifti.header['pixdim'][3]
        cm_in_voxels = int(np.floor(10/reference_axial_dim))
        axial_limits = [-1, -1]
        # There is likely a more effecient and clear way to find these bounds.
        for z in xrange(reference_esophagus_numpy.shape[2]):
            if np.sum(reference_esophagus_numpy[:,:,z]) > 0:
                if axial_limits[0] == -1:
                    if z+cm_in_voxels > reference_esophagus_numpy.shape[2]:
                        axial_limits[0] = reference_esophagus_numpy.shape[2]
                    else:
                        axial_limits[0] = z+cm_in_voxels
                axial_limits[1] = z+1-cm_in_voxels
        # Crop esophagus and spinal cord segmentations accordingly.
        for crop_segmentation in [os.path.join(mount_path, str.split(InputGroundTruth_Base, '.')[0] + '_split', 'Esophagus.nii.gz'), os.path.join(mount_path, str.split(InputGroundTruth_Base, '.')[0] + '_split', 'SpinalCord.nii.gz'), os.path.join(mount_path, str.split(InputGroundTruth_Base, '.')[0] + '_split', 'Esophagus.nii.gz'), os.path.join(mount_path, str.split(InputSeg_Base, '.')[0] + '_split', 'SpinalCord.nii.gz')]:
            crop_nifti = nib.load(crop_segmentation)
            crop_numpy = crop_nifti.get_fdata()
            crop_numpy[:,:,0:int(axial_limits[0])] = 0
            crop_numpy[:,:,int(axial_limits[1]+1):] = 0
            nib.save(nib.Nifti1Image(crop_numpy, crop_nifti.affine), crop_segmentation)
    else:
        print('WARNING: No esophagus file found in the ground truth DICOM-RT. No truncation of the spinal cord to match the esophagus will occur.')

    # Create Output CSV
    output_array = np.zeros((2, len(Labels.keys())*3+1), dtype=object)
    output_array[0,0] = 'Segmentation Name'
    output_array[1,0] = InputSeg_Base

    # Grab matching labels from the test segmentation, and compute label statistics
    output_index = 1
    for Segmentation_Label in glob.glob(os.path.join(mount_path, str.split(InputSeg_Base, '.')[0] + '_split', '*')):

        Label_Name = str.split(os.path.basename(os.path.normpath(Segmentation_Label)), '.')[0]

        if Label_Name in Labels.keys():

            Docker_Command = 'docker run --rm -v ' + mount_path + ':/input_dicom mayoqin/plastimatch plastimatch dice --all ./input_dicom/' + str.split(InputGroundTruth_Base, '.')[0] + '_split/' + Label_Name + '.nii.gz ./input_dicom/' + str.split(InputSeg_Base, '.')[0] + '_split/' + Label_Name + '.nii.gz'
            output = subprocess.check_output(Docker_Command, shell=True)
            output = str.split(output, '\n')
            output = [x.replace(' ', '') for x in output]

            print(output)

            DICE = str.split(output[7], ':')[1]
            HAUSDORFF = str.split(output[-2], '=')[1]
            AVERAGE_DISTANCE = str.split(output[-4], '=')[1]

            output_array[0,output_index] = Label_Name + '_DICE'
            output_array[0,output_index + 1] = Label_Name + '_HAUSDORFF'
            output_array[0,output_index + 2] = Label_Name + '_AVERAGE_DISTANCE'            
            output_array[1,output_index] = DICE
            output_array[1,output_index + 1] = HAUSDORFF
            output_array[1,output_index + 2] = AVERAGE_DISTANCE
            output_index += 3

            print(os.path.basename(os.path.join(mount_path, str.split(InputSeg_Base, '.')[0] + '_split', Label_Name + '.nii.gz')))
            print(DICE, ' ', HAUSDORFF, ' ', AVERAGE_DISTANCE)

            Labels.pop(Label_Name)

    # Note any labels in Ground Truth that the input segmentation failed to segment.
    for key in Labels:
        output_array[0,output_index] = key + '_DICE'
        output_array[0,output_index + 1] = key + '_HAUSDORFF'
        output_array[0,output_index + 2] = key + '_AVERAGE_DISTANCE'
        output_array[1,output_index] = 'NA'
        output_array[1,output_index + 1] = 'NA'
        output_array[1,output_index + 2] = 'NA'
        output_index += 3

    # Save output.
    with open(OutputSheet, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        for row in output_array:
            csvfile.writerow(row)

    # Optional: remove segmentation directory upon completion.
    # rmtree(mount_path)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--seg", dest="InputSeg", help="Segmentation to Compare (DICOM Directory)")
    parser.add_option("--groundtruth", dest="InputGroundTruth", help="Ground Truth to Compare (DICOM Directory)")
    parser.add_option("--volume", dest="InputVolume", help="CT Volume to Compare (DICOM Directory)")
    parser.add_option("--output_dir", dest="OutputDir", help="Statistics File to Output To (CSV File)")
    parser.add_option("--output_csv", dest="OutputSheet", help="Statistics File to Output To (CSV File)")
    (options, args) = parser.parse_args()
    Compare_Segmentations(options.InputSeg, options.InputGroundTruth, options.InputVolume, options.OutputDir, options.OutputSheet)
