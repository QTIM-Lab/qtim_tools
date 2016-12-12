from ..qtim_utilities import nifti_util

import numpy as np
import nibabel as nib
import math
import os

def convert_intensity_to_concentration(data_numpy, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=0, T1_map = []):

	flip_angle_radians = flip_angle_degrees*np.pi/180

	if T1_map != []:
		R1_pre = 1 / T1_map
		R1_pre = np.reshape(R1_pre.shape + (1,))
	elif T1_blood == 0:
		R1_pre = 1 / T1_tissue
	else:
		R1_pre = 1 / T1_blood

	a = np.exp(-1 * TR * R1_pre)
	relative_term = (1-a) / (1-a*np.cos(flip_angle_radians))

	# Note-to-self: Find a way for this to work regardless of array dimension.

	if len(data_numpy.shape) == 1:
		baseline = np.mean(data_numpy[0:int(np.round(injection_start_time_seconds/time_interval_seconds))])
		baseline = np.tile(baseline, data_numpy.shape[-1])
	if len(data_numpy.shape) == 2:
		baseline = np.mean(data_numpy[:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=1)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0], 1)), (1,data_numpy.shape[-1]))
	if len(data_numpy.shape) == 3:
		baseline = np.mean(data_numpy[:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=2)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0],baseline.shape[1], 1)), (1,1,data_numpy.shape[-1]))
	if len(data_numpy.shape) == 4:
		baseline = np.mean(data_numpy[:,:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=3)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0],baseline.shape[1],baseline.shape[2], 1)), (1,1,1,data_numpy.shape[-1]))
	
	output_numpy = np.copy(data_numpy)

	output_numpy = output_numpy / baseline
	output_numpy = output_numpy * relative_term

	output_numpy = (output_numpy - 1) / (a * (output_numpy * np.cos(flip_angle_radians) - 1))
	output_numpy = -1 * (1 / (relaxivity * TR)) * np.log(output_numpy)

	output_numpy = np.nan_to_num(output_numpy)

	if T1_blood == 0:
		return output_numpy
	else:
		output_numpy = output_numpy / (1-hematocrit)
		return output_numpy

def create_4d_from_3d(filepath, stacks=5):

	""" Mainly to make something work with NordicICE. TO-DO: Make work with anything but the Tofts phantom.
		Also move to nifti_util when that is finished.
	"""

	nifti_3d = nib.load(filepath)
	numpy_3d = nifti_3d.get_data()
	numpy_4d = np.zeros((numpy_3d.shape[0], numpy_3d.shape[1], stacks, numpy_3d.shape[2]), dtype=float)
	numpy_3d = np.reshape(numpy_3d, (numpy_3d.shape[0], numpy_3d.shape[1], 1, numpy_3d.shape[2]))

	numpy_4d = np.tile(numpy_3d, (1,1,stacks,1))

	t1_map = np.zeros((numpy_3d.shape[0], numpy_3d.shape[1], stacks), dtype=float)
	t1_map[0:50,0:70,:] = 1000
	print t1_map
	t1_map[t1_map == 0] = 1440
	print t1_map

	# t1_map = np.reshape(t1_map, (t1_map.shape[0], t1_map.shape[1], 1, t1_map.shape[2]))
	# t1_map = np.tile(t1_map, (1,1,5,1))

	nifti_util.save_numpy_2_nifti(numpy_4d, filepath, 'tofts_4d.nii')
	nifti_util.save_numpy_2_nifti(t1_map, filepath, 'tofts_t1map.nii')

def create_gradient_phantom(filepath, label_filepath):

	""" TO-DO: Fix the ktrans variation so that it correctly ends in 0.35, instead of whatever
		it currently ends in. Also parameterize and generalize everything for more interesting
		phantoms.
	"""

	nifti_3d = nib.load(filepath)
	numpy_3d = nifti_3d.get_data()

	label_3d = nib.load(label_filepath).get_data()
	AIF_label_value = 1
	dimension = 3

	AIF_subregion = np.copy(numpy_3d)
	label_mask = (label_3d[:,:,0] != AIF_label_value).reshape((label_3d.shape[0:-1] + (1,)))
	AIF_subregion = np.ma.array(AIF_subregion, mask=np.tile(label_mask, (1,)*(dimension-1) + (AIF_subregion.shape[-1],)))
	AIF_subregion = np.reshape(AIF_subregion, (np.product(AIF_subregion.shape[0:-1]), AIF_subregion.shape[-1]))
	AIF = AIF_subregion.mean(axis=0, dtype=np.float64)

	time_interval_seconds = float((11*60) / numpy_3d.shape[-1])
	time_series = np.arange(0, AIF.size) / (60 / time_interval_seconds)
	
	contrast_AIF = generate_contrast_agent_concentration(AIF,T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=1440)

	gradient_nifti = numpy_3d[:,:,0:2].astype(float)
	gradient_nifti[:,0:10,:] = 0
	gradient_nifti[:,70:,:] = 0
	time_nifti = np.copy(numpy_3d).astype(float)

	for ve_idx, ve in enumerate(np.arange(.01, .5 +.5/50, .5/50)):
		for ktrans_idx, ktrans in enumerate(np.arange(.01, .35 +.35/60, .35/60)):
			gradient_nifti[ve_idx, ktrans_idx+10, 0] = float(ktrans)
			gradient_nifti[ve_idx, ktrans_idx+10, 1] = float(ve)
			time_nifti[ve_idx, ktrans_idx+10,:] = estimate_concentration([np.log(ktrans),-1 * np.log((1-ve)/ve)], contrast_AIF, time_series)
			print np.shape(time_nifti)
			print [ve_idx, ktrans_idx]
			print [ve, ktrans]

	nifti_util.save_numpy_2_nifti(time_nifti, filepath, 'gradient_toftsv6_concentration')
	time_nifti[:,10:70,:] = revert_concentration_to_intensity(data_numpy=time_nifti[:,10:70,:], reference_data_numpy=numpy_3d[:,10:70,:], T1_tissue=1000, TR=5, flip_angle_degrees=30, injection_start_time_seconds=60, relaxivity=.0045, time_interval_seconds=time_interval_seconds, hematocrit=.45, T1_blood=0, T1_map = [])

	nifti_util.save_numpy_2_nifti(gradient_nifti[:,:,0], filepath, 'gradient_toftsv6_ktrans_truth')
	nifti_util.save_numpy_2_nifti(gradient_nifti[:,:,1], filepath, 'gradient_toftsv6_ve_truth')
	nifti_util.save_numpy_2_nifti(time_nifti, filepath, 'gradient_toftsv6')

def revert_concentration_to_intensity(data_numpy, reference_data_numpy, T1_tissue, TR, flip_angle_degrees, injection_start_time_seconds, relaxivity, time_interval_seconds, hematocrit, T1_blood=0, T1_map = []):

	if T1_map != []:
		R1_pre = 1 / T1_map
		R1_pre = np.reshape(R1_pre.shape + (1,))
	else:
		R1_pre = 1 / T1_tissue

	flip_angle_radians = flip_angle_degrees*np.pi/180
	a = np.exp(-1 * TR * R1_pre)
	relative_term = (1-a) / (1-a*np.cos(flip_angle_radians))

	if len(reference_data_numpy.shape) == 1:
		baseline = np.mean(reference_data_numpy[0:int(np.round(injection_start_time_seconds/time_interval_seconds))])
		baseline = np.tile(baseline, reference_data_numpy.shape[-1])
	if len(reference_data_numpy.shape) == 2:
		baseline = np.mean(reference_data_numpy[:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=1)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0], 1)), (1,reference_data_numpy.shape[-1]))
	if len(reference_data_numpy.shape) == 3:
		baseline = np.mean(reference_data_numpy[:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=2)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0],baseline.shape[1], 1)), (1,1,reference_data_numpy.shape[-1]))
	if len(reference_data_numpy.shape) == 4:
		baseline = np.mean(reference_data_numpy[:,:,:,0:int(np.round(injection_start_time_seconds/time_interval_seconds))], axis=3)
		baseline = np.tile(np.reshape(baseline, (baseline.shape[0],baseline.shape[1],baseline.shape[2], 1)), (1,1,1,reference_data_numpy.shape[-1]))

	data_numpy = np.exp(data_numpy / (-1 / (relaxivity*TR)))
	data_numpy = (data_numpy * a -1) / (data_numpy * a * np.cos(flip_angle_radians) - 1)
	data_numpy = data_numpy / relative_term
	data_numpy = data_numpy * baseline
	###

	return data_numpy

if __name__ == "__main__":
	pass