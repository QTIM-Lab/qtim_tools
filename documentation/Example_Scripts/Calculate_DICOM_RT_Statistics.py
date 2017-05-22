import os
import glob
import numpy as np
import subprocess
import csv

from optparse import OptionParser
from subprocess import call
from shutil import copytree, rmtree

def Compare_Segmentations(InputSeg, InputGroundTruth, InputVolume, OutputSheet):

    for filename in [InputSeg, InputGroundTruth, InputVolume]:
        filename = os.path.abspath(filename)

    InputSeg_Base = os.path.basename(os.path.normpath(InputSeg)).replace(" ", "")
    InputGroundTruth_Base = os.path.basename(os.path.normpath(InputGroundTruth)).replace(" ", "")
    InputVolume_Base = os.path.basename(os.path.normpath(InputVolume)).replace(" ", "")

    mount_path = os.path.abspath('./output_dicom')

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
    for label_idx, Ground_Truth_Label in enumerate(glob.glob('./output_dicom/' + str.split(InputGroundTruth_Base, '.')[0] + '_split/*')):
        Label_Name = str.split(os.path.basename(os.path.normpath(Ground_Truth_Label)), '.')[0]
        Labels[Label_Name] = label_idx

    # Create Output CSV
    output_array = np.zeros((2, len(Labels.keys())*2+1), dtype=object)
    output_array[0,0] = 'Segmentation Name'
    output_array[1,0] = InputSeg_Base

    # Grab matching labels from the test segmentation, and compute label statistics
    output_index = 1
    for Segmentation_Label in glob.glob('./output_dicom/' + str.split(InputSeg_Base, '.')[0] + '_split/*'):

        Label_Name = str.split(os.path.basename(os.path.normpath(Segmentation_Label)), '.')[0]

        if Label_Name in Labels.keys():

            Docker_Command = 'docker run --rm -v ' + mount_path + ':/input_dicom mayoqin/plastimatch plastimatch dice --all ./input_dicom/' + str.split(InputGroundTruth_Base, '.')[0] + '_split/' + Label_Name + '.nii.gz ./input_dicom/' + str.split(InputSeg_Base, '.')[0] + '_split/' + Label_Name + '.nii.gz'
            output = subprocess.check_output(Docker_Command, shell=True)
            output = str.split(output, '\n')
            output = [x.replace(' ', '') for x in output]

            print output

            DICE = str.split(output[7], ':')[1]
            HAUSDORFF = str.split(output[-2], '=')[1]

            output_array[0,output_index] = Label_Name + '_DICE'
            output_array[0,output_index + 1] = Label_Name + '_HAUSDORFF'
            output_array[1,output_index] = DICE
            output_array[1,output_index + 1] = HAUSDORFF
            output_index += 2

            print DICE, ' ', HAUSDORFF

            Labels.pop(Label_Name)

    # Note any labels in Ground Truth that the input segmentation failed to segment.
    for key in Labels:
        output_array[0,output_index] = key + '_DICE'
        output_array[0,output_index + 1] = key + '_HAUSDORFF'
        output_array[1,output_index] = 'NA'
        output_array[1,output_index + 1] = 'NA'
        output_index += 2

    # Save output.
    with open(OutputSheet, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        for row in output_array:
            csvfile.writerow(row)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--seg", dest="InputSeg", help="Segmentation to Compare (DICOM Directory)")
    parser.add_option("-g", "--groundtruth", dest="InputGroundTruth", help="Ground Truth to Compare (DICOM Directory)")
    parser.add_option("-v", "--volume", dest="InputVolume", help="CT Volume to Compare (DICOM Directory)")
    parser.add_option("-o", "--output", dest="OutputSheet", help="Statistics File to Output To (CSV File)")
    (options, args) = parser.parse_args()
    Compare_Segmentations(options.InputSeg, options.InputGroundTruth, options.InputVolume, options.OutputSheet)