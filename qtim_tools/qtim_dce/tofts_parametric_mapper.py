""" TODO: Fill in docstring (sorry...)

"""

from __future__ import division

# TODO Clean up these imports..
# Import nifti_util. It only fails if this is script is not loaded as a package, which it should be.

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti, nifti_2_numpy
from qtim_tools.qtim_dce.dce_util import convert_intensity_to_concentration, parker_model_AIF, generate_AIF

from functools import partial
import numpy as np
import nibabel as nib
import scipy.optimize
import scipy.ndimage
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import math
import os
import time
import fnmatch
import csv

from queue import Queue
from threading import Thread
from multiprocessing.pool import Pool
from multiprocessing import freeze_support

def calc_DCE_properties_single(filepath, T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_file=[], label_suffix=[], label_value=1, mask_value=0, mask_threshold=0, T1_map_file=[], T1_map_suffix='-T1Map', AIF_label_file=[],  AIF_value_data=[], AIF_value_suffix=[], convert_AIF_values=True, AIF_mode='label_average', AIF_label_suffix=[], AIF_label_value=1, label_mode='separate', param_file=[], default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outputs=['ktrans','ve','auc'], outfile_prefix='', processes=1, gaussian_blur=.65, gaussian_blur_axis=2):

    """ This is a master function that creates ktrans, ve, and auc values from raw intensity 1D-4D volumes.
    """

    print('\n')

    # NaN values are cleaned for ease of calculation.
    if isinstance(filepath, str):
        image = np.nan_to_num(nifti_2_numpy(filepath))
    else:
        image = np.nan_to_num(np.copy(filepath))

    # Unlikely that this circumstance will apply to 4D images in the future..
    dimension = len(image.shape)
    if dimension > 4:
        print('Error: Images greater than dimension 4 are currently not supported. Skipping this volume...')
        return []

    # Convenience variables created from input parameters.
    flip_angle_radians = flip_angle_degrees*np.pi/180
    time_interval_seconds = float(scan_time_seconds / image.shape[dimension-1])
    timepoints = image.shape[-1]
    bolus_time = int(np.ceil((injection_start_time_seconds / scan_time_seconds) * timepoints))

    # This step applies a gaussian blur to the provided axes. Blurring greatly increases DCE accuracy on noisy data.
    image = preprocess_dce(image, gaussian_blur=gaussian_blur, gaussian_blur_axis=gaussian_blur_axis)

    # Data store in other files may be relevant to DCE calculations. This utility function collects them and stores them in local variables.
    AIF_label_image, label_image, T1_image, AIF = retreive_data_from_files(filepath, label_file, label_mode, label_suffix, label_value, AIF_label_file, AIF_label_value, AIF_mode, AIF_label_suffix, T1_map_file, T1_map_suffix, AIF_value_data, AIF_value_suffix, image)

    # If no pre-set AIF text file is provided, one must be generated either from a label-map or a population AIF.
    if AIF == []:
        AIF = generate_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image, AIF_label_image, AIF_value_data, AIF_mode, dimension, AIF_label_value)

    # Error-catching for broken AIFs.
    if AIF == []:
        print('Problem calculating AIF. Skipping this volume...')
        return []

    # Signal conversion is required in order for raw data to interface with the Tofts model.
    contrast_image = convert_intensity_to_concentration(image, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit)

    # Depending on where the AIF is derived from, AIF values may also need to be convered into Gd concentration.
    if AIF_mode == 'population':
        contrast_AIF = AIF
    elif AIF_value_data != [] and convert_AIF_values == False:
        contrast_AIF = AIF
    else:
        contrast_AIF = convert_intensity_to_concentration(AIF, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=T1_blood)

    # The optimization portion of the program is run here.
    parameter_maps = simplex_optimize(contrast_image, contrast_AIF, time_interval_seconds, bolus_time, image, label_image, mask_value, mask_threshold, initial_fitting_function_parameters, outputs, processes)

    # Outputs are saved, and then returned.
    for param_idx, param in enumerate(outputs):
        save_numpy_2_nifti(parameter_maps[...,param_idx], filepath, outfile_prefix + param + '.nii.gz')
    return outputs

def retreive_data_from_files(filepath, label_file, label_mode, label_suffix, label_value, AIF_label_file, AIF_label_value, AIF_mode, AIF_label_suffix, T1_map_file, T1_map_suffix, AIF_value_data, AIF_value_suffix, image=[]):

    """ Check for associated data relevant to DCE calculation either via provided filenames or provided filename suffixes (e.g. '-T1map')
    """

    # TODO: Clean up redundancies in this function.

    # Check for a provided region of interest for calculating parameter values.
    if label_mode == 'none':
        label_image = []
    elif label_file != []:
        label_image = nifti_2_numpy(label_file)
    elif label_suffix != []:
        split_path = str.split(filepath, '.nii')
        if os.path.isfile(split_path[0] + label_suffix + '.nii' + split_path[1]):
            label_image = nifti_2_numpy(split_path[0] + label_suffix + '.nii' + split_path[1])
        elif os.path.isfile(split_path[0] + AIF_label_suffix + '.nii.gz'):
            label_image = nifti_2_numpy(split_path[0] + label_suffix + '.nii.gz')
        else:
            print("No labelmap found at provided label suffix. Continuing without...")
            label_image = []
    else:
        label_image = []

    # Check for a provided region of interest for determining an AIF.
    if AIF_mode == 'label_average':
        if AIF_label_file != []:
            AIF_label_image = nifti_2_numpy(AIF_label_file)
        elif AIF_label_suffix != []:
            split_path = str.split(filepath, '.nii')
            if os.path.isfile(split_path[0] + AIF_label_suffix + '.nii' + split_path[1]):
                AIF_label_image = nifti_2_numpy(split_path[0] + AIF_label_suffix + '.nii' + split_path[1])
            elif os.path.isfile(split_path[0] + AIF_label_suffix + '.nii.gz'):
                AIF_label_image = nifti_2_numpy(split_path[0] + AIF_label_suffix + '.nii.gz')
            else:
                print("No AIF labelmap found at provided label suffix. Continuing without...")
                AIF_label_image = []
        elif label_mode == 'separate':
            print('No label found for this AIF. If AIF label is in the same file as ROI, change the label_mode parameter to \'combined\'. Skipping this volume...')
            AIF_label_image = []
        elif label_mode == 'combined':
            if label_file != []:
                AIF_label_image = np.copy(label_image)
                AIF_label_image[label_image != AIF_label_value] = 0
            else:
                print('No label found for this AIF. If the AIF label is in a separate file from the ROI, change the label_mode parameter to \'separate\'. If not, be sure that the AIF_label_value parameter matches the AIF label value in your ROI. Skipping this volume...')
                AIF_label_image = []
    elif AIF_mode == 'population':
        AIF_label_image = []
    else:
        print('Invalid AIF_mode parameter. This volume will be skipped. \n')
        AIF_label_image = []

    # Check for a provided T1 mapping file, which will be relevant to signal conversion.
    if T1_map_file != []:
        T1_image = nifti_2_numpy(T1_map_file)
    elif T1_map_suffix != []:
        split_path = str.split(filepath, '.nii')
        if os.path.isfile(split_path[0] + T1_map_suffix + '.nii' + split_path[1]):
            T1_image = nifti_2_numpy(split_path[0] + T1_map_suffix + '.nii' + split_path[1])
        elif os.path.isfile(split_path[0] + T1_map_suffix + '.nii.gz' + split_path[1]):
            T1_image = nifti_2_numpy(split_path[0] + T1_map_suffix + '.nii.gz' + split_path[1])
        else:
            T1_image = []
            print('No T1 map found at provided T1 map file suffix. Continuing without... \n')       
    else:
        T1_image = []

    if T1_image != [] and (image.shape[0:-1] != T1_image.shape):
        print('T1 map and DCE image are not the same shape. T1 map processing will be skipped. \n')
        T1_image = []


    # This option is for text files that have AIF values in either raw or signal-converted format.
    # TODO: Address different delimiters between files? Or maybe others have to do this.
    if AIF_value_data != []:
        if AIF_value_suffix != []:
            split_path = str.split(filepath, '.nii')
            if os.path.isfile(split_path[0] + AIF_value_suffix + '.txt'):
                try:
                    AIF = np.loadtxt(split_path[0] + AIF_value_suffix + '.txt', dtype=object, delimiter=';')
                    AIF = [value for value in AIF if value != '']

                    if len(AIF) != image.shape[-1]:
                        print('AIF does not have as many timepoints as image. Assuming AIF timepoints are post-baseline, and filling pre-baseline points with zeros. \n')
                        new_AIF = np.zeros(image.shape[-1], dtype=float)
                        new_AIF[-len(AIF):] = AIF
                        AIF = new_AIF
                except:
                    print("Error reading AIF values file. AIF reader requires text files with semicolons (;) as delimiters. Skipping this volume... \n")
                    AIF = []
            else:
                AIF_value_data = []
                print('No AIF values found at provided AIF value suffix. Continuing without... \n')   
        if isinstance(AIF_value_data, str):
            try:
                AIF = np.loadtxt(AIF_value_data, dtype=object, delimiter=';')
                AIF = [value for value in AIF if value != '']

                if len(AIF) != image.shape[-1]:
                    print('AIF does not have as many timepoints as image. Assuming AIF timepoints are post-baseline, and filling pre-baseline points with zeros. \n')
                    new_AIF = np.zeros(image.shape[-1], dtype=float)
                    new_AIF[-len(AIF):] = AIF
                    AIF = new_AIF
            except:
                print("Error reading AIF values file. AIF reader requires text files with semicolons (;) as delimiters. Skipping this volume... \n")
                AIF = []
        elif AIF_value_data != []:
            AIF = AIF_value_data
        else:
            AIF = []
    else:
        AIF = []

    return AIF_label_image, label_image, T1_image, AIF

def preprocess_dce(image_numpy=[], gaussian_blur=0, gaussian_blur_axis=-1):

    """ Apply pre-processing methods to incoming DCE data. Currently, only gaussian blurring is available, although
        other methods might be available in the future. Gaussian blurring greatly improves fitting performance on
        phantom data and increases repeatability on patient data.
    """

    if gaussian_blur > 0:

        output_numpy = np.copy(image_numpy)
        dims = len(image_numpy.shape)

        if gaussian_blur > 0:

            # check_image(output_numpy[...,0], mode="maximal_slice")
            blur_axes = [gaussian_blur]*(dims-1) + [0]

            if gaussian_blur_axis > 0:
                blur_axes[gaussian_blur_axis] = 0 

            output_numpy = scipy.ndimage.filters.gaussian_filter(image_numpy, blur_axes)     
            # check_image(output_numpy[...,0], mode="maximal_slice")

        return output_numpy

    else:
        return image_numpy

def simplex_optimize(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, bolus_time, image=[], label_image=[], mask_value=0, mask_threshold=0, initial_fitting_function_parameters=[.01,.1], outputs=['ktrans','ve'], processes=1):

    """ This function sets up parallel processing. Until this function is reimplemented in Cython, paralell processing
        may be required to get semi-normal processing speeds.
    """

    # I am extremely skeptical about this broken masking method.
    if label_image != []:
        contrast_image_numpy[label_image == 0] = mask_value

    if image != []:
        contrast_image_numpy[image[...,0] <= mask_threshold] = mask_value
        contrast_image_numpy[image[...,0] == mask_value] = mask_value

    if processes > 1:
        subunits = []
        sublength = np.floor(contrast_image_numpy.shape[0] / processes)

        print('Dividing data into ' + str(processes) + ' subgroups of length.. ' + str(int(sublength)) + ' units.')

        for i in range(processes - 1):
            subunits += [contrast_image_numpy[int(i*sublength):int((i+1)*sublength),...]]

        subunits += [contrast_image_numpy[int((processes - 1)*sublength):,...]]

        subprocess = partial(simplex_optimize_loop, contrast_AIF_numpy=contrast_AIF_numpy, time_interval_seconds=time_interval_seconds, bolus_time=bolus_time, mask_value=mask_value, mask_threshold=mask_threshold, initial_fitting_function_parameters=initial_fitting_function_parameters)

        optimization_pool = Pool(processes)
        results = optimization_pool.map(subprocess, subunits)

        output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)
        stitch_index = 0
        for result in results:
            output_image[stitch_index:stitch_index+result.shape[0],...] = result
            stitch_index += result.shape[0]

    else:
        output_image = simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, bolus_time, mask_value, mask_threshold, initial_fitting_function_parameters)

    return output_image

def simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, bolus_time, mask_value=0, mask_threshold=0, initial_fitting_function_parameters=[1,1]):

    contrast_AIF_numpy = contrast_AIF_numpy[bolus_time:]

    #np.set_printoptions(threshold=np.nan)
    power = np.power
    sum = np.sum
    e = math.e

    time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)
    time_interval = time_series[1]

    ktransmax = 1


    def cost_function(params):
        #estimated_concentration=[0]
        #append = estimated_concentration.append
        ktrans = params[0]
        ve = params [1]
        kep = ktrans / ve
        
        """log_e = -1 * kep * time_interval
        capital_E = np.exp(log_e,dtype=np.float128)
        log_e_2 = log_e ** 2

        block_A = (capital_E - log_e -1)
        block_B = (capital_E -(capital_E * log_e) -1)
        
        if np.isnan(block_B):
            block_B = 0.0

        block_ktrans = ktrans * time_interval / log_e_2

        for i in range(1,np.size(contrast_AIF_numpy)):
            term_A = contrast_AIF_numpy[i] * block_A
            term_B = contrast_AIF_numpy[i-1] * block_B
            result = estimated_concentration[-1] * capital_E + block_ktrans * (term_A - term_B)
            if mp.isnan(result) or mp.isinf(result):
                result = 0.0
            append(result)"""

        ###Faster, but potentially inaccurate

        res=np.exp(-1*kep*time_series)
        estimated_concentration=ktrans*np.convolve(contrast_AIF_numpy,res)*time_series[1]
        estimated_concentration=estimated_concentration[0:np.size(res)]

        difference_term = observed_concentration - estimated_concentration
        difference_term = np.power(difference_term,2)

        return np.sum(difference_term)




    # These constraints are currently unused.
    def ve_constraint1(params):
        return params[1] - 1e-3

    def ve_constraint2(params):
        return 1 - params[1]

    def ktrans_constraint1(params):
        return params[0] - 1e-3

    def ktrans_constraint2(params):
        return ktransmax - params[0]

    # Remember to change later if there are different amounts of outputs.
    output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)

    space_dims = contrast_image_numpy.shape[0:-1]

    for index in np.ndindex(space_dims):

        # Need to think about how to implement masking. Maybe np.ma involved. Will likely require
        # editing other np.math functions down the line.
        if contrast_image_numpy[index + (0,)] == mask_value:
            output_image[index + (0,)] = -.01
            output_image[index + (1,)] = -.01
            output_image[index + (2,)] = -.01
            continue

        # Because multiprocessing divvies up the image into pieces, these indexed values
        # do not have real-world meaning.
        # print(index)

        observed_concentration = contrast_image_numpy[index][bolus_time:]

        # I am currently unsure when to start calculating AUC. Perhaps have this specified by the user? TODO.
        auc = trapz(observed_concentration)

        # with timewith('concentration estimator') as timer:
        initial_fitting_function_parameters = [.01,.1]        
        result_params, fopt, iterations, funcalls, warnflag, allvecs = scipy.optimize.fmin(cost_function, initial_fitting_function_parameters, disp=0, ftol=1e-14, xtol=1e-8, full_output = True, retall=True)

        ktrans = result_params[0]
        ve = result_params[1]

       #print([ktrans, ve, auc])

        output_image[index + (0,)] = ktrans
        output_image[index + (1,)] = ve
        output_image[index + (2,)] = auc

    # These masking values are arbitrary and will likely differ between AIFs. TODO: Figure out a way to reconcile that.
    output_image[...,1][output_image[...,0] < .05] = 0
    output_image[...,2][abs(output_image[...,2]) > 1e6] = 0
    output_image[...,0][output_image[...,0] > .95*ktransmax] = 0
    output_image[...,1][output_image[...,1] > .99] = 0

    return output_image


def calc_DCE_properties_batch(folder, regex='', recursive=False, T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_file=[], label_suffix=[], label_value=1, mask_value=0, mask_threshold=0, T1_map_file=[], T1_map_suffix='-T1Map', AIF_label_file=[],  AIF_value_data=[], convert_AIF_values=True, AIF_mode='label_average', AIF_label_suffix=[], AIF_label_value=1, label_mode='separate', param_file=[], default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outputs=['ktrans','ve','auc'], outfile_prefix='', processes=1, gaussian_blur=.65, gaussian_blur_axis=2):


    """ Currently non-functional, although has not been tested since original writing. Meant to process DCE images in batch.
    """

    suffix_exclusion_regex = []
    for suffix in label_suffix, T1_map_suffix, AIF_label_suffix, AIF_value_suffix:
        if suffix != []:
            suffix_exclusion_regex += [suffix]
    
    volume_list = []
    if recursive:
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, regex):
                volume_list.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(folder):
            if fnmatch.fnmatch(regex):
                volume_list.append(os.path.join(root, filename))
    
    for volume in volume_list:
        for suffix in suffix_exclusion_regex:
            if suffix_exclusion_regex not in volume:

                print('Working on volume located at... ' + volume)

                calc_DCE_properties_single(volume, T1_tissue, T1_blood, relaxivity, TR, TE, scan_time_seconds, hematocrit, injection_start_time_seconds, flip_angle_degrees, label_file, label_suffix, label_value, mask_value, mask_threshold, T1_map_file, T1_map_suffix, AIF_label_file,  AIF_value_data, convert_AIF_values, AIF_mode, AIF_label_suffix, AIF_label_value, label_mode, param_file, default_population_AIF, initial_fitting_function_parameters, outputs, outfile_prefix, processes, gaussian_blur, gaussian_blur_axis)

def test_method_2d():
    # print('hello')
    filepath = 'C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v6.nii.gz'
    # filepath = 'C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/qtim_tools/test_data/test_data_dce/gradient_toftsv6.nii'
    # filepath = 'C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v9_5SNR.nii'
    # filepath = 'C:/Users/abeers/Documents/GitHub/Public_QTIM/qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v6.nii.gz'

    calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_value_data=[], convert_AIF_values=False, outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_suffix=[], AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outfile_prefix='tofts_v6_gradient', processes=16, mask_threshold=20, mask_value=-1, gaussian_blur=0, gaussian_blur_axis=-1)

    # calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_value_data=[], convert_AIF_values=False, outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_suffix=[], AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, initial_fitting_function_parameters=[.3,.3], outfile_prefix='tofts_v9_noblur_', processes=22, mask_threshold=-1, mask_value=-1, gaussian_blur=0, gaussian_blur_axis=-1)


def test_method_3d(filepath=[]):
    # print('hello')
    # These are params for NHX/CED data
    if filepath == []:
        filepath = 'C:/Users/azb22/Documents/Junk/dce_mc_st_corrected.nii'
        filepath = 'C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom/DCE_MRI_Phantom_RawData.nii.gz'
        AIF_filepath = 'C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom/DCE_MRI_Phantom_AIF-label.nii.gz'
        ROI = 'C:/Users/azb22/Documents/Scripting/DCE_Motion_Phantom/DCE_MRI_Phantom_RawData_ROI.nii.gz'

    AIF_value_data = 'C:/Users/azb22/Documents/Junk/VISIT_01_autoAIF_bAIF.txt'
    # calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_value_data=AIF_value_data, convert_AIF_values=False, outputs=['ktrans','ve','auc'], T1_tissue=1500, T1_blood=1440, relaxivity=.0039, TR=6.8, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=160, flip_angle_degrees=10, label_suffix=[], AIF_mode='population', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outfile_prefix='mead_cobyal_individual_ktrans_no_mask', processes=22, mask_threshold=20, mask_value=-1, gaussian_blur=.65, gaussian_blur_axis=2)

    calc_DCE_properties_single(filepath, label_file=ROI, param_file=[], AIF_label_file=AIF_filepath, AIF_value_data=[], convert_AIF_values=False, outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=3.8, TE=2.1, scan_time_seconds=195, hematocrit=0.45, injection_start_time_seconds=24, flip_angle_degrees=25, label_suffix=[], AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, initial_fitting_function_parameters=[.01,.1], outfile_prefix='mead_cobyal_individual_ktrans_no_mask', processes=22, mask_threshold=0, mask_value=-1, gaussian_blur=0, gaussian_blur_axis=2)


if __name__ == '__main__':

    # freeze_support()
    # np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    # np.set_printoptions(suppress=True, precision=4)

    # test_method_2d()
    test_method_3d()
    # create_4d_from_3d(filepath)
