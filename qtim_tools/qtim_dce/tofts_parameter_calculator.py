from __future__ import division

from ..qtim_utilities import nifti_util

import numpy as np
import nibabel as nib
import scipy.optimize
import matplotlib.pyplot as plt
import math
import random
import os
import dce_util

global_AIF = []

def calc_DCE_properties_single(filepath, T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, label_file=[], label_suffix=[], label_value=1, mask_value=0, mask_threshold=0, T1_map_file=[], T1_map_suffix='-T1Map', AIF_label_file=[],  AIF_values_file=[], AIF_mode='label_average', AIF_label_suffix=[], AIF_label_value=1, label_mode='separate', param_file=[], default_population_AIF=False, intial_fitting_function_parameters=[-2,-2], outputs=['ktrans','ve','auc'], outfile_prefix=''):

	flip_angle_radians = flip_angle_degrees*np.pi/180
	image = nifti_util.nifti_2_numpy(filepath)
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

	contrast_image = dce_util.convert_intensity_to_concentration(image, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit)
	contrast_AIF = dce_util.convert_intensity_to_concentration(AIF, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=1440)

	# Note-to-self: implement 'output' parameter functionality. Any custom-specified outputs
	# will currently break the program.
	parameter_maps = np.zeros((contrast_image.shape[0:-1] + (len(outputs),)), dtype=float)
	parameter_maps = simplex_optimize(contrast_image, contrast_AIF, time_interval_seconds, mask_value, mask_threshold, intial_fitting_function_parameters)

	for param_idx, param in enumerate(outputs):
		nifti_util.save_numpy_2_nifti(parameter_maps[:,:,param_idx], filepath, outfile_prefix + param + '.nii.gz')

def retreive_labels_from_files(filepath, label_file, label_mode, label_suffix, label_value, AIF_label_file, AIF_label_value, AIF_mode, AIF_label_suffix, T1_map_file, T1_map_suffix):

	if label_file != []:
		label_image = nifti_util.nifti_2_numpy(label_file)
	elif label_suffix != []:
		split_path = str.split(filepath, '.nii')
		if os.path.isfile(split_path[0] + label_suffix + '.nii' + split_path[1]):
			label_image = nifti_util.nifti_2_numpy(split_path[0] + label_suffix + '.nii' + split_path[1])
		else:
			print "No labelmap found at provided label suffix. Continuing without..."
			label_image = []
	else:
		label_image = []

	# Note: specific label value functionality not yet added.

	if AIF_mode == 'label_average':
		if AIF_label_file != []:
			AIF_label_image = nifti_util.nifti_2_numpy(AIF_label_file)
		elif AIF_label_suffix != []:
			split_path = str.split(filepath, '.nii')
			AIF_label_image = nifti_util.nifti_2_numpy(split_path[0] + AIF_label_suffix + '.nii' + split_path[1])
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
		T1_image = nifti_util.nifti_2_numpy(T1_map_file)
	elif T1_map_suffix != []:
		split_path = str.split(filepath, '.nii')
		if os.path.isfile(split_path[0] + T1_map_suffix + '.nii' + split_path[1]):
			T1_image = nifti_util.nifti_2_numpy(split_path[0] + T1_map_suffix + '.nii' + split_path[1])
		else:
			T1_image = []
			print 'No T1 map found at provided T1 map file suffix. Continuing without...'		
	else:
		T1_image = []

	return AIF_label_image, label_image, T1_image


def simplex_optimize(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value=0, mask_threshold=0, intial_fitting_function_parameters=[1,1]):
	
	# inexplicable minute conversion; investigate
	time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)

	def cost_function(params):

		estimated_concentration = estimate_concentration(params, contrast_AIF_numpy, time_series)

		difference_term = observed_concentration - estimated_concentration
		difference_term = np.power(difference_term, 2)

		return np.sum(difference_term)


	result_grid = np.zeros((6,5,2),dtype=float)
	np.set_printoptions(threshold=np.nan)
	output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)

	# Apologies for the mess of commented lines; these will be cleaned up soon (TO-DO)

	if len(contrast_image_numpy.shape) == 3:
		# for x in xrange(contrast_image_numpy.shape[0]):
			# for y in xrange(contrast_image_numpy.shape[1]):
		for x_idx, x in enumerate(range(0,50,1)):
			for y_idx, y in enumerate(range(10,70,1)):
				# if contrast_image_numpy[x,y,0] == mask_value or contrast_image_numpy[x,y,0] < mask_threshold:
					# continue

				print[x,y]
				observed_concentration = contrast_image_numpy[x,y,:]

				result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=1)

				ktrans = np.exp(result_params[0]) #ktrans
				ve = 1 / (1 + np.exp(-result_params[1])) #ve
				# auc = np.trapz(observed_concentration, dx=time_interval_seconds) / np.trapz(contrast_AIF_numpy, dx=time_interval_seconds)

				print [ktrans, ve]
				# result_grid[y_idx, x_idx, 0] = ktrans
				# result_grid[y_idx, x_idx, 1] = ve

				# time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)
				# estimated_concentration = estimate_concentration(result_params, contrast_AIF_numpy, time_interval_seconds)
				# plt.plot(time_series, estimated_concentration, 'r--', time_series, observed_concentration, 'b--')
				# plt.show()

				output_image[x,y,0] = ktrans
				output_image[x,y,1] = ve
				# output_image[x,y,2] = auc

		output_image[output_image[:,:,0] > .7] = -.01
		output_image[output_image[:,:,1] > .9] = -.01
		output_image[output_image[:,:,1] < 1e-4] = -.01

		validation_image = np.copy(output_image)
		validation_image = np.ma.masked_equal(validation_image, -.01)
		for y in xrange(0,50,10):
			for x in xrange(10,70,10):
				result_grid[(x)/10,(y-1)/10,0] = np.mean((validation_image[x:x+10,y:y+10,0]).flatten())
				result_grid[(x)/10,(y-1)/10,1] = np.mean((validation_image[x:x+10,y:y+10,1]).flatten())

		print result_grid[:,:,0]
		print result_grid[:,:,1]

	elif len(contrast_image_numpy.shape) == 4:
		for x in xrange(contrast_image_numpy.shape[0]):
			for y in xrange(contrast_image_numpy.shape[1]):
				for z in xrange(contrast_image_numpy.shape[2]):
					if contrast_image_numpy[x,y,z,0] == mask_value or contrast_image_numpy[x,y,z,0] < mask_threshold:
						continue
					print[x,y,z]
					print contrast_image_numpy[x,y,z,0]
					observed_concentration = contrast_image_numpy[x,y,z,:]

					result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=1, ftol=.0001, xtol=1e-8)

					ktrans = np.exp(result_params[0]) #ktrans
					ve = 1 / (1 + np.exp(-result_params[1])) #ve
					# auc = np.trapz(observed_concentration, dx=time_interval_seconds) / np.trapz(contrast_AIF_numpy, dx=time_interval_seconds)

					print [ktrans, ve]
					output_image[x,y,z,0] = ktrans
					output_image[x,y,z,1] = ve
					# output_image[x,y,z,2] = auc
		output_image[output_image[:,:,:,0] > .7] = 0
		output_image[output_image[:,:,:,1] > .9] = 0

	else:
		print "Fitting not yet implemented for images with less than 3 dimensions or greater than four dimensions."
		return []


	return output_image

def estimate_concentration(params, contrast_AIF_numpy, time_series):

	estimated_concentration = np.zeros_like(contrast_AIF_numpy, dtype=float)
	if params[0] > 10 or params[1] > 10:
		return estimated_concentration

	ktrans = np.exp(params[0])
	ve = 1 / (1 + np.exp(-params[1]))
	kep = ktrans / ve

	# Notation is very inexact here. Clean it up later.

	for i in xrange(1, np.size(estimated_concentration)):
		log_e = -1 * kep * time_series[1]
		capital_E = np.exp(log_e)
		term_A = contrast_AIF_numpy[i] * (capital_E - log_e - 1)
		term_B = contrast_AIF_numpy[i-1] * (capital_E - (capital_E * log_e) - 1)
		integral_term = (term_A - term_B) / np.power(log_e, 2)
		estimated_concentration[i] = estimated_concentration[i-1]*capital_E + ktrans * time_series[1] * integral_term

	# Quick, error prone convolution method
	# res = np.exp(-1*kep*time_series)
	# estimated_concentration = ktrans * np.convolve(contrast_AIF_numpy, res) * time_series[1]
	# estimated_concentration = estimated_concentration[0:np.size(res)]

	return estimated_concentration

def validate_math():
	return

def validate_params():
	return

def validate_label():
	return

def generate_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, image_numpy=[], AIF_label_numpy=[], AIF_mode='label_average', dimension=4, AIF_label_value=1):

	# It's an open question how to create labels for 2-D DCE phantoms. For now, I assume that people draw their label at time-point zero.

	if AIF_mode == 'label_average':
		if image_numpy != []:
			if AIF_label_numpy != []:


				# Note-to-self: Find a way for this to work regardless of array dimension.
				# Also find a better way to mask arrays.
				# Also this function is extremely messy. It will fail if an image has zero
				# values. This will probably occur, so it must be fixed soon.

				if dimension == 3:
					AIF_subregion = np.copy(image_numpy)
					label_mask = (AIF_label_numpy[:,:,0] != AIF_label_value).reshape((AIF_label_numpy.shape[0:-1] + (1,)))
					AIF_subregion = np.ma.array(AIF_subregion, mask=np.tile(label_mask, (1,)*(dimension-1) + (AIF_subregion.shape[-1],)))
					AIF_subregion = np.reshape(AIF_subregion, (np.product(AIF_subregion.shape[0:-1]), AIF_subregion.shape[-1]))
					AIF = AIF_subregion.mean(axis=0, dtype=np.float64)
					return AIF
				elif dimension == 4:
					AIF_subregion = np.copy(image_numpy)
					label_mask = (AIF_label_numpy[:,:,:] != AIF_label_value).reshape((AIF_label_numpy.shape[0:-1] + (1,)))
					AIF_subregion = np.ma.array(AIF_subregion, mask=np.tile(label_mask, (1,)*(dimension-1) + (AIF_subregion.shape[-1],)))
					AIF_subregion = np.reshape(AIF_subregion, (np.product(AIF_subregion.shape[0:-1]), AIF_subregion.shape[-1]))
					AIF = AIF_subregion.mean(axis=0, dtype=np.float64)
					return AIF_subregion
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
		print 'Population AIF mode not yet completed. Unable to calculate AIF.'
	return []



def calc_DCE_properties_batch(folder, labels=True, param_file='', outputs=['ktrans','auc','ve'], T1_tissue=1000, T1_blood=1660, relaxivity=.0039, TR=5, TE=2.1, slice_time_interval=.5, hematocrit=0.4, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='population', AIF_label_suffix='-label-AIF', AIF_label_value=1):

	return

def test_method():
	filepath = 'C:/Users/azb22/Documents/GitHub/Public_qtim_tools/qtim_tools/test_data/test_data_dce/tofts_v9_5SNR.nii'
	calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[-2,.1], outfile_prefix='tofts_v9_5SNR')

if __name__ == '__main__':


	np.set_printoptions(suppress=True, precision=4)
	test_method()

	# label_filepath = 'tofts_v6-AIF-label.nii.gz'

	# filepath = 'tofts_v6.nii.gz'
	# filepath = 'tofts_v9_5SNR.nii'
	# filepath = 'dce1_mc_ss.nii.gz'

	# create_gradient_phantom(filepath, label_filepath)

	# calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[.1,.1])
	# calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(6*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[-2,.1], outfile_prefix='tofts_v9_5SNR')
	# calc_DCE_properties_single(filepath, label_file=[], param_file=[], AIF_label_file=[], AIF_values_file=[], outputs=['ktrans','ve','auc'], T1_tissue=1000, T1_blood=1440, relaxivity=.0045, TR=5, TE=2.1, scan_time_seconds=(11*60), hematocrit=0.45, injection_start_time_seconds=60, flip_angle_degrees=30, AIF_mode='label_average', AIF_label_suffix='-AIF-label', AIF_label_value=1, label_mode='separate', default_population_AIF=False, intial_fitting_function_parameters=[1,1])

	# create_4d_from_3d(filepath)