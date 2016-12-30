from __future__ import division

# from ..qtim_utilities import nifti_util
from functools import partial
# from scipy

import numpy as np
import nibabel as nib
import scipy.optimize
import matplotlib.pyplot as plt
import math
import random
import os
# import dce_util
import re
import time

from Queue import Queue
from threading import Thread
from multiprocessing.pool import Pool

class timewith():
    def __init__(self, name=''):
        self.name = name
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start

    def checkpoint(self, name=''):
        print '{timer} {checkpoint} took {elapsed} seconds'.format(
            timer=self.name,
            checkpoint=name,
            elapsed=self.elapsed,
        ).strip()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.checkpoint('finished')
        pass

def nifti_2_numpy(filepath):

    """ There are a lot of repetitive conversions in the current iteration
        of this program. Another option would be to always pass the nibabel
        numpy class, which contains image and attributes. But not everyone
        knows how to use that class, so it may be more difficult to troubleshoot.
    """

    img = nib.load(filepath).get_data().astype(float)
    return img

def save_numpy_2_nifti(image_numpy, reference_nifti_filepath, output_path):
    nifti_image = nib.load(reference_nifti_filepath)
    image_affine = nifti_image.affine
    output_nifti = nib.Nifti1Image(image_numpy, image_affine)
    nib.save(output_nifti, output_path)

def calc_DCE_properties_single(filepath, T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_file=[], label_suffix=[], label_value=1, mask_value=0, mask_threshold=0, T1_map_file=[], T1_map_suffix='-T1Map', AIF_label_file=[],  AIF_values_file=[], AIF_mode='label_average', AIF_label_suffix=[], AIF_label_value=1, label_mode='separate', param_file=[], default_population_AIF=False, intial_fitting_function_parameters=[.3,.3], outputs=['ktrans','ve','auc'], outfile_prefix='', processes=1):

    print '\n'

    flip_angle_radians = flip_angle_degrees*np.pi/180

    # Is there any plausible benefit to keeping NaN values?
    image = np.nan_to_num(nifti_2_numpy(filepath))

    dimension = len(image.shape)
    if dimension > 4:
        print 'Error: Images greater than dimension 4 are currently not supported. Skipping this volume...'
        return []

    time_interval_seconds = float(scan_time_seconds / image.shape[dimension-1])

    AIF_label_image, label_image, T1_image = retreive_labels_from_files(filepath, label_file, label_mode, label_suffix, label_value, AIF_label_file, AIF_label_value, AIF_mode, AIF_label_suffix, T1_map_file, T1_map_suffix)

    if T1_image != [] and (image.shape[0:-1] != T1_image.shape):
        print 'T1 map and DCE image are not the same shape. T1 map processing will be skipped.'
        T1_image = []

    AIF = generate_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image, AIF_label_image, AIF_mode, dimension, AIF_label_value)

    if AIF == []:
        print 'Problem calculating AIF. Skipping this volume...'
        return []

    contrast_image = convert_intensity_to_concentration(image, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit)

    if AIF_mode != 'population':
        contrast_AIF = convert_intensity_to_concentration(AIF, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=T1_blood)
    else:
        contrast_AIF = AIF

    # Note-to-self: implement 'output' parameter functionality. Any custom-specified outputs
    # will currently break the program.
    parameter_maps = simplex_optimize(contrast_image, contrast_AIF, time_interval_seconds, mask_value, mask_threshold, intial_fitting_function_parameters, outputs, processes)

    for param_idx, param in enumerate(outputs):
        save_numpy_2_nifti(parameter_maps[...,param_idx], filepath, outfile_prefix + param + '.nii.gz')

def retreive_labels_from_files(filepath, label_file, label_mode, label_suffix, label_value, AIF_label_file, AIF_label_value, AIF_mode, AIF_label_suffix, T1_map_file, T1_map_suffix):

    if label_file != []:
        label_image = nifti_2_numpy(label_file)
    elif label_suffix != []:
        split_path = str.split(filepath, '.nii')
        if os.path.isfile(split_path[0] + label_suffix + '.nii' + split_path[1]):
            label_image = nifti_2_numpy(split_path[0] + label_suffix + '.nii' + split_path[1])
        elif os.path.isfile(split_path[0] + AIF_label_suffix + '.nii.gz'):
            label_image = nifti_2_numpy(split_path[0] + label_suffix + '.nii.gz')
        else:
            print "No labelmap found at provided label suffix. Continuing without..."
            label_image = []
    else:
        label_image = []

    # Note: specific label value functionality not yet added.

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
                print "No AIF labelmap found at provided label suffix. Continuing without..."
                AIF_label_image = []
        elif label_mode == 'separate':
            print 'No label found for this AIF. If AIF label is in the same file as ROI, change the label_mode parameter to \'combined\'. Skipping this volume...'
            AIF_label_image = []
        elif label_mode == 'combined':
            if label_file != []:
                AIF_label_image = np.copy(label_image)
                AIF_label_image[label_image != AIF_label_value] = 0
            else:
                print 'No label found for this AIF. If the AIF label is in a separate file from the ROI, change the label_mode parameter to \'separate\'. If not, be sure that the AIF_label_value parameter matches the AIF label value in your ROI. Skipping this volume...'
                AIF_label_image = []

    elif AIF_mode == 'population':
        AIF_label_image = []

    if T1_map_file != []:
        T1_image = nifti_2_numpy(T1_map_file)
    elif T1_map_suffix != []:
        split_path = str.split(filepath, '.nii')
        if os.path.isfile(split_path[0] + T1_map_suffix + '.nii' + split_path[1]):
            T1_image = nifti_2_numpy(split_path[0] + T1_map_suffix + '.nii' + split_path[1])
        else:
            T1_image = []
            print 'No T1 map found at provided T1 map file suffix. Continuing without...'       
    else:
        T1_image = []

    return AIF_label_image, label_image, T1_image

# def preprocess_dce(image_numpy=[], gaussian_blur=0, gaussian_blur_axis=-1):

#     if gaussian_blur > 0:

#         if gaussian_blur_axis > 0:


#         else:

#             image_numpy = 



def generate_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image_numpy=[], AIF_label_numpy=[], AIF_mode='label_average', dimension=4, AIF_label_value=1):

    # It's an open question how to create labels for 2-D DCE phantoms. For now, I assume that people draw their label at time-point zero.

    if AIF_mode == 'label_average':
        if image_numpy != []:
            if AIF_label_numpy != []:


                # Note-to-self: Maybe find a way for this to work regardless of array dimension.
                # Also find a better way to mask arrays.
                # Broadcasting would be nice, if it worked as intended. Would solve the memory overhead here,
                # although it's hard to imagine it mattering (super-high resolution images?) (Parallel batch processing?)
                # Also this function is extremely messy and unreadable. It will fail if an image has zero
                # values. This will probably occur, so it must be fixed soon.

                AIF_subregion = np.nan_to_num(np.copy(image_numpy))

                if dimension == 3:
                    # Acquiring label mask...
                    label_mask = (AIF_label_numpy[:,:,0] != AIF_label_value)
                    # Reshaped for array broadcasting purposes...
                    label_mask = label_mask.reshape((AIF_label_numpy.shape[0:-1] + (1,)))
                    # Making use of numpy's confusing array tiling dynamic to mask all time points with the label...
                    masked_AIF_subregion = np.ma.array(AIF_subregion, mask=np.tile(label_mask, (1,)*(dimension-1) + (AIF_subregion.shape[-1],)))
                    # Reshaping for ease of calculating the mean...
                    masked_AIF_subregion = np.reshape(masked_AIF_subregion, (np.product(masked_AIF_subregion.shape[0:-1]), masked_AIF_subregion.shape[-1]))
                    AIF = masked_AIF_subregion.mean(axis=0, dtype=np.float64)
                    return AIF
                elif dimension == 4:
                    label_mask = (AIF_label_numpy != AIF_label_value)
                    broadcast_label_mask = np.repeat(label_mask[:,:,:,np.newaxis], AIF_subregion.shape[-1], axis=3)
                    masked_AIF_subregion = np.ma.masked_array(AIF_subregion, mask=broadcast_label_mask)             
                    masked_AIF_subregion = np.reshape(masked_AIF_subregion, (np.product(masked_AIF_subregion.shape[0:-1]), masked_AIF_subregion.shape[-1]))
                    AIF = masked_AIF_subregion.mean(axis=0, dtype=np.float64)
                    return AIF
                else:
                    print 'Error: too many or too few dimensions to calculate AIF currently. Unable to calculate AIF.'
                    return []
            else:
                'Error: no AIF label detected. Unable to calculate AIF.'
                return []
        else:
            print 'No image provided to AIF function. Set AIF_mode to \'population\' to use a population AIF. Unable to calculate AIF.'
            return []

    elif AIF_mode == 'population':
        AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image_numpy)
        return AIF

    return []

def parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image_numpy=[]):

    timepoints = image_numpy.shape[-1]
    AIF = np.zeros(timepoints)

    bolus_time = np.ceil((injection_start_time_seconds / scan_time_seconds) * timepoints)

    time_series_minutes = time_interval_seconds * np.arange(timepoints-bolus_time) / 60

    # Parker parameters. Taken from his orginal published paper.
    a1 = 0.809
    a2 = 0.330
    T1 = 0.17406
    T2 = 0.365
    sigma1 = 0.0563
    sigma2 = 0.132
    alpha = 1.050
    beta = 0.1685
    s = 38.078
    tau = 0.483

    # This is taken from our original Matlab script. I don't know the Matlab script well
    # enough to make more descriptive variable names.
    term_0 = alpha*np.exp(-1 * beta * time_series_minutes) / (1 + np.exp(-s*(time_series_minutes-tau)))
    
    A1 = a1 / (sigma1 * ((2*np.pi)**.5))
    B1 = np.exp(-(time_series_minutes-T1)**2 / (2*sigma1**2))
    term_1 = A1 * B1

    A2 = a2 / (sigma2 * ((2*np.pi)**.5))
    B2 = np.exp(-(time_series_minutes-T2)**2 / (2*sigma2**2))
    term_2 = A2 * B2

    post_bolus_AIF = term_0 + term_1 + term_2

    AIF[bolus_time:] = post_bolus_AIF

    return AIF

def convert_intensity_to_concentration(data_numpy, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=0, T1_map = []):

    flip_angle_radians = flip_angle_degrees*np.pi/180

    if T1_map != []:
        R1_pre = float(1) / float(T1_map)
        R1_pre = np.reshape(R1_pre.shape + (1,))
    elif T1_blood == 0:
        R1_pre = float(1) / float(T1_tissue)
    else:
        R1_pre = float(1) / float(T1_blood)

    a = np.exp(-1 * TR * R1_pre)
    relative_term = (1-a) / (1-a*np.cos(flip_angle_radians))

    # Tuple notation is very confusing, but I unfortunately have to use it to ge this working with
    # arbitrary dimensions.

    dim = len(data_numpy.shape)

    if dim == 1:
        baseline = np.mean(data_numpy[0:int(np.round(injection_start_time_seconds/time_interval_seconds))])
        baseline = np.tile(baseline, data_numpy.shape[-1])
    elif dim > 1 and dim < 5:
        baseline = np.mean(data_numpy[...,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=dim-1)
        baseline = np.tile(np.reshape(baseline, (baseline.shape[0:dim-1] + (1,))), (1,)*(dim-1) + (data_numpy.shape[-1],))
    else:
        print 'Dimension error. Please enter an array with dimensions between 1 and 4.'

    output_numpy = np.copy(data_numpy)

    output_numpy = np.nan_to_num(output_numpy / baseline)

    output_numpy = output_numpy * relative_term

    output_numpy = (output_numpy - 1) / (a * (output_numpy * np.cos(flip_angle_radians) - 1))

    output_numpy[output_numpy < 0] = 0

    output_numpy = -1 * (1 / (relaxivity * TR)) * np.log(output_numpy)

    output_numpy = np.nan_to_num(output_numpy)

    if T1_blood == 0:
        return output_numpy
    else:
        output_numpy = output_numpy / (1-hematocrit)
        return output_numpy

def simplex_optimize(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value=0, mask_threshold=0, intial_fitting_function_parameters=[.3,.3], outputs=['ktrans','ve'], processes=1):

    # Parallelization using multiprocessing has been tricky. Relative imports don't seem
    # to work with multiprocessing.. Maybe I should get out of Python 2.7 - it seems like a lot of
    # multiprocessing was built for 3.

    if processes > 1:
        subunits = []
        sublength = np.floor(contrast_image_numpy.shape[0] / processes)

        for i in xrange(processes - 1):
            subunits += [contrast_image_numpy[int(i*sublength):int((i+1)*sublength),...]]

        subunits += [contrast_image_numpy[int((processes - 1)*sublength):,...]]

        subprocess = partial(simplex_optimize_loop, contrast_AIF_numpy=contrast_AIF_numpy, time_interval_seconds=time_interval_seconds, mask_value=0, mask_threshold=0, intial_fitting_function_parameters=[1,1])

        optimization_pool = Pool(processes)
        results = optimization_pool.map(subprocess, subunits)

        output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)
        stitch_index = 0
        for result in results:
            output_image[stitch_index:stitch_index+result.shape[0],...] = result
            stitch_index += result.shape[0]

    else:
        output_image = simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value, mask_threshold, intial_fitting_function_parameters)

    return output_image

def simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value=0, mask_threshold=0, intial_fitting_function_parameters=[.3,.3]):

    # return np.zeros_like(contrast_image_numpy)

    mask_threshold = -1000

    np.set_printoptions(threshold=np.nan)
    power = np.power
    sum = np.sum
    e = math.e

    # inexplicable minute conversion; investigate changing whole optimization to seconds. It looks like the Tofts model is just set up that way.
    time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)
    time_interval = time_series[1]

    def cost_function(params):

        # The estimate concentration function is repeated locally to eke out every last bit of efficiency
        # from this massively looping program. As much as possible is calculated outside the loop for
        # performance reasons. Appending is faster than pre-allocating space in this case - who knew.

        estimated_concentration = [0]

        append = estimated_concentration.append

        ktrans = params[0]
        ve = params[1]
        kep = ktrans / ve

        log_e = -1 * kep * time_interval
        capital_E = e**log_e
        log_e_2 = log_e**2

        block_A = (capital_E - log_e - 1)
        block_B = (capital_E - (capital_E * log_e) - 1)
        block_ktrans = ktrans * time_interval / log_e_2

        for i in xrange(1, np.size(contrast_AIF_numpy)):
            term_A = contrast_AIF_numpy[i] * block_A
            term_B = contrast_AIF_numpy[i-1] * block_B
            append(estimated_concentration[-1]*capital_E + block_ktrans * (term_A - term_B))

        difference_term = observed_concentration - estimated_concentration
        difference_term = power(difference_term, 2)

        return sum(difference_term)

    # Remember to change later if different amounts of outputs.
    output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)
    # If there's an easy, readable way to make this code the same for 3 and 4 dimensions, I haven't found it yet.
    # Iterating over all-but-one dimensions has made nd-iter tricky for me.

    if len(contrast_image_numpy.shape) == 3:
        for x in xrange(contrast_image_numpy.shape[0]):
            for y in xrange(contrast_image_numpy.shape[1]):

                # Need to think about how to implement masking. Maybe np.ma involved. Will likely require
                # editing other np.math functions down the line.
                if contrast_image_numpy[x,y,z,0] < mask_threshold:
                    output_image[x,y,0] = -.01
                    output_image[x,y,1] = -.01
                    continue
                # Because multiprocessing divvies up the image into pieces, these x and y values
                # do not have real-world meaning.
                # print[x,y]

                observed_concentration = contrast_image_numpy[x,y,:]
                
                with timewith('concentration estimator') as timer:
                    result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=0, ftol=.001, xtol=1e-4)

                # This weird parameter transform is a holdover from Xiao's program. I wonder what purpose it serves..
                # ktrans = np.exp(result_params[0]) #ktrans
                # ve = 1 / (1 + np.exp(-result_params[1])) #ve

                ktrans = result_params[0]
                ve = result_params[1]

                # I am currently unsure how to calculate AUC. Perhaps have this specified by the user? TODO.
                # auc = np.trapz(observed_concentration, dx=time_interval_seconds) / np.trapz(contrast_AIF_numpy, dx=time_interval_seconds)

                # These cutoffs were arbitrarily determined. Would be interesting to find cutoffs with a
                # a biological basis, or better yet, stop these functions from exploding in the first place..
                if ktrans > .6 or ve < .01 or ktrans < .01 or ve > .9:
                    intial_fitting_function_parameters = [.3, .3]
                else:
                    intial_fitting_function_parameters = [ktrans, ve]

                print [ktrans, ve]
                output_image[x,y,0] = ktrans
                output_image[x,y,1] = ve
                # output_image[x,y,2] = auc

                # This plotting snippet currently not working, modify later..
                # time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)
                # estimated_concentration = estimate_concentration(result_params, contrast_AIF_numpy, time_interval)
                # plt.plot(time_series, estimated_concentration, 'r--', time_series, observed_concentration, 'b--')
                # plt.show()

    elif len(contrast_image_numpy.shape) == 4:
        for x in xrange(contrast_image_numpy.shape[0]):
            for y in xrange(contrast_image_numpy.shape[1]):
                for z in xrange(contrast_image_numpy.shape[2]):

                    # if contrast_image_numpy[x,y,z,0] == mask_value or contrast_image_numpy[x,y,z,0] < mask_threshold:
                    #     continue

                    if contrast_image_numpy[x,y,z,0] < mask_threshold:
                        output_image[x,y,z,0] = -.01
                        output_image[x,y,z,1] = -.01
                        continue

                    print[x,y,z]

                    observed_concentration = contrast_image_numpy[x,y,z,:]

                    with timewith('concentration estimator') as timer:
                        result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=0, ftol=.001, xtol=1e-4)

                    ktrans = result_params[0]
                    ve = result_params[1]

                    if ktrans > .6 or ve < .01 or ktrans < .01 or ve > .9:
                        intial_fitting_function_parameters = [.3, .3]
                    else:
                        intial_fitting_function_parameters = [ktrans, ve]

                    print [ktrans, ve]
                    output_image[x,y,z,0] = ktrans
                    output_image[x,y,z,1] = ve
                    # output_image[x,y,z,2] = auc

    else:
        print "Fitting not yet implemented for images with fewer than 3 dimensions or greater than four dimensions."
        return []

    output_image[output_image[...,0] > 2] = -.01
    # output_image[output_image[...,1] > .99] = -.01
    # output_image[output_image[...,1] < 1e-4] = -.01


    return output_image

    return []

def calc_DCE_properties_batch(folder, labels=True, param_file='', outputs=['ktrans','auc','ve'], T1_tissue=1000, T1_blood=1660, relaxivity=.0039, TR=5, TE=2.1, slice_time_interval=.5, hematocrit=0.4, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='population', AIF_label_suffix='-label-AIF', AIF_label_value=1):

    return

def test_method_2d():
    # print 'hello'
    filepath = 'C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v6.nii.gz'
    calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[.3,.3], outfile_prefix='tofts_v6', processes=1)

def test_method_3d():
    # print 'hello'
    # These are params for NHX/CED data
    filepath = 'C:/Users/azb22/Documents/Junk/dce_mc_st_corrected.nii'
    calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1500, T1_blood=1440, relaxivity=.0039, TR=6.8, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=160, flip_angle_degrees=10, AIF_mode='population', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[.3,.3], outfile_prefix='tofts_v6', processes=12)

if __name__ == '__main__':

    # print 'hello'
    # pass

    # np.set_printoptions(suppress=True, precision=4, threshold=np.nan)
    np.set_printoptions(suppress=True, precision=4)
    # test_method_2d()
    test_method_3d()

    # label_filepath = 'tofts_v6-AIF-label.nii.gz'

    # filepath = 'tofts_v6.nii.gz'
    # filepath = 'tofts_v9_5SNR.nii'
    # filepath = 'dce1_mc_ss.nii.gz'

    # create_gradient_phantom(filepath, label_filepath)

    # calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[.1,.1])
    # calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[-2,.1], outfile_prefix='tofts_v9_5SNR')
    # calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[1,1])

    # create_4d_from_3d(filepath)