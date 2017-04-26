import numpy as np
import os

from qtim_tools.qtim_utilities.array_util import generate_rotation_affine, save_affine, get_jacobian_determinant, return_jacobian_matrix, generate_identity_affine
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti, save_numpy_2_nifti_no_reference

from scipy.io import savemat
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter

from subprocess import call

def Load_4D_DICOM_to_NRRD(directory='', Slicer_path=''):


import os
import glob

from optparse import OptionParser

def convert_dicom(input_folder, output_filename):
    
    dicom_files = glob.glob(os.path.join(input_folder, '*'))

    # db = slicer.dicomDatabase

    if slicer.dicomDatabase == None:
        slicer.dicomDatabase = ctk.ctkDICOMDatabase()
        slicer.dicomDatabase.openDatabase(os.path.dirname(os.path.realpath(__file__)) + "Slicer_Dicom_Database/ctkDICOM.sql", "SLICER")
    db = slicer.dicomDatabase

    plugins = [slicer.modules.dicomPlugins['DICOMScalarVolumePlugin'](), slicer.modules.dicomPlugins['DICOMScalarVolumePlugin'](), slicer.modules.dicomPlugins['DICOMScalarVolumePlugin']()]

    for plugin in plugins:
        print plugin
        # try:
        if plugin:
            loadables = plugin.examine([dicom_files])

            if len(loadables) == 0:
                print('plugin failed to interpret this series')
            else:

                patientID = db.fileValue(loadables[0].files[0],'0010,0020')
                seriesDescription = db.fileValue(loadables[0].files[0],'0008,103e')
                seriesDescription = "".join(x for x in seriesDescription if x.isalnum())
                seriesDate = db.fileValue(loadables[0].files[0],'0008,0020')
                seriesTime = db.fileValue(loadables[0].files[0],'0008,0031')
                flipAngle = db.fileValue(loadables[0].files[0],'0018,1314')
                echoTime = db.fileValue(loadables[0].files[0],'0018,0081')
                repTime = db.fileValue(loadables[0].files[0],'0018,0080')

                output_directory = os.path.dirname(output_filename)
                output_filename =  os.path.join(output_directory, patientID + '_' + seriesDescription + '.nii.gz')

                print output_filename
                print os.getcwd()

                volume = plugin.load(loadables[0])
                if volume:
                    slicer.util.saveNode(volume,output_filename)
                    slicer.util.quit()
                    return

        else:
            continue

        # except:
            # continue

    slicer.util.quit()
    return

def Slicer_Rotate(input_numpy, reference_nifti, affine_matrix, Slicer_path="/opt/Slicer-4.5.0-1-linux-amd64/Slicer"):

    save_numpy_2_nifti(input_numpy, reference_nifti, 'temp.nii.gz')
    save_affine(affine_matrix, 'temp.txt')

    Slicer_Command = [Slicer_path, '--launch', 'ResampleScalarVectorDWIVolume', 'temp.nii.gz', 'temp_out.nii.gz', '-f', 'temp.txt', '-i', 'bs']

    call(' '.join(Slicer_Command), shell=True)

    return convert_input_2_numpy('temp_out.nii.gz')

def Slicer_PkModeling(input_nrrd, Slicer_path="Slicer"):

    Slicer_Command = [Slicer_path, '--launch', 'PkModeling', 'temp.nii.gz', 'temp_out.nii.gz', '-f', 'temp.txt', '-i', 'bs']

    return

def Generate_Head_Jerk(input_filepath, timepoint, duration, rotation_peaks=[3, 3, 0], input_filepath='', reference_nifti='', output_filepath=''):

    input_numpy = convert_input_2_numpy(input_filepath)

    if reference_nifti == '':
        reference_nifti = input_filepath

    endpoint = timepoint + duration
    midpoint = timepoint + np.round(endpoint - timepoint)/2
    rotation_matrix_increment = np.array([float(x)/float(timepoint-endpoint) for x in rotation_peaks])

    if endpoint > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer than the duration of the volume'

    rotation_direction = np.array([0,0,0])

    for t in xrange(input_numpy.shape[-1]):
        if t > timepoint and t < endpoint:
            
            if t > midpoint:
                rotation_direction = rotation_direction - rotation_matrix_increment
            if t <= midpoint:
                rotation_direction = rotation_direction + rotation_matrix_increment

            current_rotation_matrix = generate_identity_affine()

            for axis, value in enumerate(rotation_direction):
                current_rotation_matrix = np.matmul(current_rotation_matrix, generate_rotation_affine(axis, value))

            input_numpy[..., t] = Slicer_Rotate(input_numpy[..., t], reference_nifti, current_rotation_matrix)

    save_numpy_2_nifti(input_numpy, reference_nifti, output_filepath)

    return

def Generate_Head_Tilt(timepoint, duration, rotation_peaks=[3, 3, 0], input_filepath='', reference_nifti='', output_filepath=''):

    input_numpy = convert_input_2_numpy(input_filepath)

    if reference_nifti == '':
        reference_nifti = input_filepath

    endpoint = timepoint + duration
    rotation_matrix_increment = np.array([float(x)/float(timepoint-endpoint) for x in rotation_peaks])
    if endpoint > input_numpy.shape[-1]:
        print 'Invalid timepoint, longer than the duration of the volume'

    rotation_direction = np.array([0,0,0])

    for t in xrange(input_numpy.shape[-1]):
        if t > timepoint:
            if t < endpoint:
                rotation_direction = rotation_direction + rotation_matrix_increment

            current_rotation_matrix = generate_identity_affine()

            for axis, value in enumerate(rotation_direction):
                current_rotation_matrix = np.matmul(current_rotation_matrix, generate_rotation_affine(axis, value))

            print current_rotation_matrix
            print t

            input_numpy[..., t] = Slicer_Rotate(input_numpy[..., t], reference_nifti, current_rotation_matrix)

    save_numpy_2_nifti(input_numpy, reference_nifti, output_filepath)

    return

def Add_White_Noise(input_filepath, output_filepath, noise_scale=1, noise_multiplier=10):

    input_numpy = convert_input_2_numpy(input_filepath)

    for t in xrange(input_numpy.shape[-1]):
        input_numpy[..., t] = input_numpy[..., t] + np.random.normal(scale=noise_scale, size=input_numpy[..., t].shape).reshape(input_numpy[..., t].shape) * noise_multiplier

    save_numpy_2_nifti(input_numpy, input_filepath, output_filepath)

def rescale(x,min_value,value_range):
    min_x = np.min(x)
    y = min_value+value_range*(x-min_x)/np.ptp(x)
    return y

def Generate_Deformable_Motion(input_dimensions = (3,3,4), output_dimensions = (100,100,16), output_filepath="/home/abeers/Projects/DCE_Motion_Phantom/Deformable_Matrix", time_points = 65, deformation_scale=1):

    # Set the degree to which your original matrix should be upsampled.
    # Ideally, the input_dimensions should cleanly divide the output_dimensions.
    zoom_ratio = []
    for i in xrange(len(input_dimensions)):
        zoom_ratio += [output_dimensions[i] // input_dimensions[i]]

    Deformable_Matrix = np.zeros((input_dimensions + (3,)), dtype=float)
    Final_Deformation_Matrix = np.zeros((output_dimensions + (3,time_points)), dtype=float)

    for t in xrange(time_points):

        # Random Initialization of Deformations
        # Calibrates to have a maximum of +/- 1mm displacement on sample DCE-MRIs
        a, b = 0*deformation_scale, 5*deformation_scale
        Deformable_Matrix[...,0:2] = (b - a) * np.random.sample(input_dimensions + (2,)) + a

        c, d = 0*deformation_scale, 1*deformation_scale
        Deformable_Matrix[...,2] = (d - c) * np.random.sample(input_dimensions) + c

        # Uses a function from our utility package, qtim_tools, to get a corresponding
        # matrix of Jacobian values at distance 1.
        # Jacobian_Matrix = get_jacobian_determinant(Deformable_Matrix)
        Jacobian_Matrix = np.gradient(np.cumsum(Deformable_Matrix[...,0], axis=0))[0]

        print np.cumsum(Deformable_Matrix[...,0], axis=0).shape
        print Jacobian_Matrix[0].shape
        print (Jacobian_Matrix < 0).sum()

        while (Jacobian_Matrix < 0).sum() > 0:

            # While any Jacobians are negative, iterate through all indices of the matrix
            # and randomly adjust values until Jacobians of each index and its neighbors
            # are zero. With a certain subset of neighbors, this will be impossible, and
            # so the process bails out of certain indices and is allowed to restart after
            # a pre-defined number of iterations.
            for index in np.ndindex(Jacobian_Matrix.shape):

                negative_jacobians = True

                surrounding_indices = [index, [index[0]-1,index[1],index[2]],[index[0]+1,index[1],index[2]],[index[0],index[1]-1,index[2]],[index[0],index[1]+1,index[2]],[index[0],index[1],index[2]-1],[index[0],index[1],index[2]+1]]
                
                iteration = 0
                while negative_jacobians:

                    # Jacobian_Matrix = get_jacobian_determinant(Deformable_Matrix)
                    Jacobian_Matrix = np.gradient(np.cumsum(Deformable_Matrix[...,0], axis=0))[0]

                    negative_jacobians = False
                    for indice in surrounding_indices:
                        try:
                            if Jacobian_Matrix[indice] < 0:
                                negative_jacobians = True
                        except:
                            pass

                    if negative_jacobians:
                        Deformable_Matrix[index[0], index[1], index[2], :] = [(b - a) * np.random.sample() + a, (b - a) * np.random.sample() + a, (d - c) * np.random.sample() + c]
                    else:
                        break

                    iteration += 1
                    if iteration == 10:
                        print Jacobian_Matrix
                        print index
                        break

            print (Jacobian_Matrix < 0).sum()

        Deformable_Matrix = rescale(Deformable_Matrix, -5, 10)
        Jacobian_Matrix = np.gradient(np.cumsum(Deformable_Matrix[...,0], axis=0))[0]
        print (Jacobian_Matrix < 0).sum()

        # Upsample matrix
        Large_Deformable_Matrix = zoom(Deformable_Matrix, zoom_ratio + [1], order=1)
        # Deformable_Matrix = Large_Deformable_Matrix 

        # Blur matrix
        Large_Deformable_Matrix[...,0:2] = gaussian_filter(Large_Deformable_Matrix[...,0:2], sigma=1)
        Large_Deformable_Matrix[...,2] = gaussian_filter(Large_Deformable_Matrix[...,2], sigma=1)

        print 'SAVING MATRIX TIMEPOINT ', t
        Final_Deformation_Matrix[0:Large_Deformable_Matrix.shape[0],0:Large_Deformable_Matrix.shape[1],0:Large_Deformable_Matrix.shape[2],:,t] = Large_Deformable_Matrix



    # Output is currently saved to Matlab, where I use imwarp to apply the deformation field (it's very fast!)
    output_dict = {}
    output_dict['deformation_matrix'] = Final_Deformation_Matrix

    savemat(output_filepath, output_dict)

    return

if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    for noise_types in [['low', 5],['mid', 10],['high', 20]]:
        for timepoint in [8, 15]:
    #             Generate_Head_Jerk(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '_Head_Jerk_frame_' + str(timepoint) + '.nii.gz',  rotation_peaks=[4, 4, 0], timepoint=timepoint, duration=6, reference_nifti='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Ktrans_Map.nii.gz')
                Generate_Head_Tilt(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '_Head_Tilt_frame_' + str(timepoint) + '.nii.gz',  rotation_peaks=[4, 4, 0], timepoint=timepoint, duration=6, reference_nifti='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Ktrans_Map.nii.gz')

    # Generate_Deformable_Motion(time_points=1)

    # for noise_types in [['low', 5],['mid', 10],['high', 20]]:
        # Add_White_Noise(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', noise_multiplier=noise_types[1])
    # for noise_types in [['lowest', .25],['low', .5],['mid', 1],['high', 2]]:
        # Generate_Deformable_Motion(output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/Deformable_Matrix_' + noise_types[0],  deformation_scale=noise_types[1])
