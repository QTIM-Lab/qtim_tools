import numpy as np
import os
import glob

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti, save_numpy_2_nifti_no_reference, create_4d_nifti_from_3d
from qtim_tools.qtim_utilities.file_util import grab_files_recursive
from qtim_tools.qtim_dce.dce_util import parker_model_AIF, estimate_concentration, revert_concentration_to_intensity, generate_AIF, convert_intensity_to_concentration
from qtim_tools.qtim_utilties.transform_util import generate_motion_jerk, generate_motion_tilt, compose_affines, generate_identity_affine, apply_affine

from scipy.io import savemat
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt

from subprocess import call
from shutil import move

def Convert_DICOM_to_NRRD(input_folder, output_folder):

    for case in glob.glob(input_folder + '/*/'):

        Slicer_Command = "/opt/Slicer-4.5.0-1-linux-amd64/Slicer --no-main-window --disable-cli-modules --python-script /home/abeers/Github/qtim_tools/qtim_tools/qtim_slicer/convert_dicom.py -i \"" + case + "\" -o \"" + output_folder + "\" -p \'MultiVolumeImporterPlugin\'"

        print Slicer_Command

        call(Slicer_Command, shell=True)

def Convert_NRRD_to_Nifti(input_4d_nrrd, reference_nifti):

    input_4d_numpy = convert_input_2_numpy(input_4d_nrrd)
    reference_nifti = reference_nifti

    create_4d_nifti_from_3d(input_4d_numpy, reference_nifti, os.path.splitext(input_4d_nrrd)[0] + '.nii.gz')

    return os.path.splitext(input_4d_nrrd)[0] + '.nii.gz'

def Slicer_PkModeling(input_folder, Slicer_path="/opt/Slicer-4.5.0-1-linux-amd64/Slicer"):

    # os.chdir('C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom')
    # input_folder = '.'
    # Slicer_path = 'C:/Users/azb22/Documents/Software/SlicerNightly/Slicer_4.6.0/Slicer.exe'

    Slicer_Command = Slicer_path + ' --launch'

    T1Blood = '--T1Blood 1440'
    T1Tissue = '--T1Tissue 1000'
    relaxivity = '--relaxivity .0045'
    hematocrit = '--hematocrit .45'
    BAT_mode = '--BATCalculationMode UseConstantBAT'
    BAT_arrival = '--constantBAT 8'
    aif_mask_command = '--aifMask '
    roi_mask_command = '--roiMask '
    t1_map_command = '--T1Map '

    # image_list = glob.glob(input_folder + '/*.nrrd')
    image_list = os.listdir(input_folder)

    for nrrd_image in image_list:

        if '.nrrd' in nrrd_image:

            move(nrrd_image, nrrd_image.replace(' ', ''))
            nrrd_image = nrrd_image.replace(' ', '')

            output_ktrans_image = str.split(nrrd_image, '.')[0] + '_ktrans.nii.gz'
            output_ve_image = str.split(nrrd_image, '.')[0] + '_ve.nii.gz'

            output_ktrans_command = '--outputKtrans ' + output_ktrans_image
            output_ve_command = '--outputVe ' + output_ve_image

            PkModeling_command = ' '.join([Slicer_Command, 'PkModeling', nrrd_image, T1Blood, T1Tissue, relaxivity, hematocrit, BAT_mode, BAT_arrival, '--usePopAif', output_ve_command, output_ktrans_command])

            # call(PkModeling_command, shell=True)

            ktrans_array = convert_input_2_numpy(output_ktrans_image)
            ve_array = convert_input_2_numpy(output_ve_image)

            for z in xrange(ktrans_array.shape[-1]):
                ktrans_array[..., z] = medfilt(ktrans_array[..., z], [3,3])
                ve_array[..., z] = medfilt(ve_array[..., z], [3,3])

            ktrans_array[ve_array == 0] = .001
            ve_array[ve_array == 0] = .001

            save_numpy_2_nifti(ktrans_array, output_ktrans_image, output_ktrans_image)
            save_numpy_2_nifti(ve_array, output_ve_image, output_ve_image)

    return

def Create_Ideal_DCE(input_folder, output_filepath = '', input_aif=''):

    input_DCEs = []
    input_niis = glob.glob(os.path.join(input_folder, '*nrrd'))

    for nii in input_niis:
        if 'ktrans' in nii or 've' in nii:
            continue
        else:
            input_DCEs += [nii]

    for input_4d_nifti in input_DCEs:

        print 'Regenerating... ', input_4d_nifti

        # if output_filepath == '':
        output_filepath = str.split(input_4d_nifti, '.')[0]

        input_ktrans = output_filepath + '_ktrans.nii.gz'
        input_ve = output_filepath + '_ve.nii.gz'

        input_4d_nifti = Convert_NRRD_to_Nifti(input_4d_nifti, input_ktrans)

        input_numpy_4d = convert_input_2_numpy(input_4d_nifti)
        output_numpy_4d = np.zeros_like(input_numpy_4d)
        input_numpy_ktrans = convert_input_2_numpy(input_ktrans)
        input_numpy_ve = convert_input_2_numpy(input_ve)

        baseline_numpy = np.mean(input_numpy_4d[..., 0:7], axis=3)

        scan_time_seconds = 307.2
        time_interval_seconds = float((scan_time_seconds) / input_numpy_4d.shape[-1])
        time_interval_minutes = time_interval_seconds/60
        time_series = np.arange(0, input_numpy_4d.shape[-1]) / (60 / time_interval_seconds)
        injection_start_time_seconds=38.4

        T1_tissue=1000
        T1_blood=1440
        TR=3.8
        flip_angle_degrees=25
        relaxivity=.0045
        hematocrit=.45

        if input_aif == '':
            population_AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, input_numpy_4d)
            concentration_AIF = population_AIF
        else:
            print 'extracting AIF...'
            AIF_label_numpy = convert_input_2_numpy(input_aif)
            AIF = generate_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, input_numpy_4d, AIF_label_numpy)
            concentration_AIF = convert_intensity_to_concentration(AIF, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=T1_blood)

        for index in np.ndindex(input_numpy_ktrans.shape):

            output_numpy_4d[index] = np.array(estimate_concentration([input_numpy_ktrans[index],input_numpy_ve[index]], concentration_AIF, time_interval_minutes))

        # Or load presaved..
        # output_numpy_4d = nifti_2_numpy('DCE_MRI_Phantom_Regenerated_Concentrations.nii.gz')

        save_numpy_2_nifti(output_numpy_4d, input_4d_nifti, output_filepath + '_Regenerated_Concentrations.nii.gz')

        output_numpy_4d = revert_concentration_to_intensity(data_numpy=output_numpy_4d, reference_data_numpy=input_numpy_4d, T1_tissue=T1_tissue, TR=TR, flip_angle_degrees=flip_angle_degrees, injection_start_time_seconds=injection_start_time_seconds, relaxivity=relaxivity, time_interval_seconds=time_interval_seconds, hematocrit=hematocrit, T1_blood=0, T1_map = [])

        save_numpy_2_nifti(output_numpy_4d, input_4d_nifti, output_filepath + '_Regenerated_Signal.nii.gz')

    return

def Add_White_Noise(input_folder, noise_scale=1, noise_multiplier=10):

    input_niis = glob.glob(os.path.join(input_folder, '*Signal.nii*'))

    for input_4d_nifti in input_niis:

        input_numpy = convert_input_2_numpy(input_4d_nifti)

        for t in xrange(input_numpy.shape[-1]):
            input_numpy[..., t] = input_numpy[..., t] + np.random.normal(scale=noise_scale, size=input_numpy[..., t].shape).reshape(input_numpy[..., t].shape) * noise_multiplier

        save_numpy_2_nifti(input_numpy, input_4d_nifti, str.split(input_4d_nifti, '.')[0] + '_noise_' + str(noise_multiplier) +'.nii.gz')

def Add_Head_Jerks(input_folder, random_rotations=5, random_duration_range=[4,9], random_rotation_peaks=[[-4,4],[-4,4],[-4,4]], durations=7, timepoints=7, rotation_peaks=[4, 4, 0],):

    input_niis = glob.glob(os.path.join(input_folder, '*Signal*noise'))
    input_niis = [x for x in input_niis if 'jerk' not in x]

    for input_4d_nifti in input_niis:

        input_4d_numpy = convert_input_2_numpy(input_4d_nifti)
        ouput_motion_array = generate_identity_affine(input_4d_numpy.shape[0])

        if random_rotations > 0:

            total_jerk_windows = []

            for random_rotation in xrange(random_rotations):

                # Will hang if more random_rotations are specified than can fit in available timepoints.
                overlapping = True
                while overlapping:
                    random_duration = np.random.randint(random_duration_range*)
                    random_timepoint = np.random.randint(0, input_4d_numpy.shape[0]-duration)
                    random_jerk_window = np.arange(random_timepoint, random_timepoint + random_duration)
                    if not any(x in total_jerk_windows for x in random_jerk_window):
                        overlapping = False
                        total_jerk_windows.extend(random_jerk_window)

                random_motion = generate_motion_jerk(duration=random_duration, timepoint=random_timepoint, rotation_peaks=[np.randint(random_rotation_peaks[0]*),np.randint(random_rotation_peaks[1]*),np.randint(random_rotation_peaks[2]*)])

                output_motion_array = compose_affines(ouput_motion_array, random_motion)

            output_4d_numpy = apply_affine(input_4d_numpy, output_motion_array, Slicer_path="C:/Users/azb22/Documents/Software/SlicerNightly/Slicer_4.6.0/Slicer.exe")

        else:
            pass

        save_numpy_2_nifti(output_4d_numpy, input_4d_nifti, str.split(input_4d_nifti, '.')[0] + '_jerk.nii.gz')


def Slicer_Rotate(input_numpy, reference_nifti, affine_matrix, Slicer_path="/opt/Slicer-4.5.0-1-linux-amd64/Slicer"):

    save_numpy_2_nifti(input_numpy, reference_nifti, 'temp.nii.gz')
    save_affine(affine_matrix, 'temp.txt')

    Slicer_Command = [Slicer_path, '--launch', 'ResampleScalarVectorDWIVolume', 'temp.nii.gz', 'temp_out.nii.gz', '-f', 'temp.txt', '-i', 'bs']

    call(' '.join(Slicer_Command), shell=True)

    return convert_input_2_numpy('temp_out.nii.gz')

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

    Add_Head_Jerks(input_folder="C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom/RIDER_DATA")

    # Slicer_PkModeling(input_folder="/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA/")
    # Create_Ideal_DCE(input_folder="C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom/RIDER_DATA")

    # Convert_DICOM_to_NRRD(input_folder="/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA/RIDER NEURO MRI", output_folder="/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA/")

    # for noise_types in [['low', 5],['mid', 10],['high', 20]]:
        # for timepoint in [8, 15]:
    #             Generate_Head_Jerk(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '_Head_Jerk_frame_' + str(timepoint) + '.nii.gz',  rotation_peaks=[4, 4, 0], timepoint=timepoint, duration=6, reference_nifti='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Ktrans_Map.nii.gz')
                # Generate_Head_Tilt(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '.nii.gz', output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Regenerated_Signal_noise_' + noise_types[0] + '_Head_Tilt_frame_' + str(timepoint) + '.nii.gz',  rotation_peaks=[4, 4, 0], timepoint=timepoint, duration=6, reference_nifti='/home/abeers/Projects/DCE_Motion_Phantom/DCE_MRI_Phantom_Ktrans_Map.nii.gz')

    # Generate_Deformable_Motion(time_points=1)

    # for noise_types in [['low', 5],['mid', 10],['high', 20]]:
    #     Add_White_Noise(input_filepath='/home/abeers/Projects/DCE_Motion_Phantom/RIDER_DATA', noise_multiplier=noise_types[1])
    # for noise_types in [['lowest', .25],['low', .5],['mid', 1],['high', 2]]:
        # Generate_Deformable_Motion(output_filepath='/home/abeers/Projects/DCE_Motion_Phantom/Deformable_Matrix_' + noise_types[0],  deformation_scale=noise_types[1])
