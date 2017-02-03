from ..qtim_utilities import nifti_util

import numpy as np
import nibabel as nib
import math
import os

# class ParamWorker(Thread):
#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue
#         self.__output_image = []

#     def run(self):
#         while True:
#            # Get the work from the queue and expand the tuple
#             contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value, mask_threshold, intial_fitting_function_parameters = self.queue.get()
#             self.__output_image = simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value, mask_threshold, intial_fitting_function_parameters)
#             self.queue.task_done()
    
#     def join(self):
#         Thread.join(self)
#         return self.__output_image

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
	t1_map[t1_map == 0] = 1440

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

def estimate_concentration(params, contrast_AIF_numpy, time_interval):

    # Notation is very inexact here. Clean it up later.

    estimated_concentration = [0]
    # if params[0] > 10 or params[1] > 10:
    #   return estimated_concentration

    append = estimated_concentration.append
    e = math.e

    ktrans = e**params[0]
    ve = 1 / (1 + e**(-params[1]))
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

    # Quick, error prone convolution method
    # print estimated_concentration
        # res = np.exp(-1*kep*time_series)
        # estimated_concentration = ktrans * np.convolve(contrast_AIF_numpy, res) * time_series[1]
        # estimated_concentration = estimated_concentration[0:np.size(res)]

    return estimated_concentration

    # # Gratuitous plotting snippet for sanity checks
    # if True and (ve > 0.2):
    # # if False and index[0] == 0 and index[1] > 11:

    #     # optimization_path = np.zeros((len(allvecs), 2), dtype=float)
    #     # for a_idx, allvec in enumerate(allvecs):
    #     #     optimization_path[a_idx, :] = allvec
    #     #     print allvec

    #     # time_series = np.arange(0, contrast_AIF_numpy.size)
    #     # estimated_concentration = estimate_concentration(result_params, contrast_AIF_numpy, time_interval)

    #     # difference_term = observed_concentration - estimated_concentration
    #     # # print sum(power(difference_term, 2))
    #     # print [ktrans, ve]
    #     # plt.plot(time_series, estimated_concentration, 'r--', time_series, observed_concentration, 'b--')
    #     # plt.show()

    #     # time_series = np.arange(0, contrast_AIF_numpy.size)
    #     # estimated_concentration = estimate_concentration([.1, .01], contrast_AIF_numpy, time_interval)

    #     # time_series = np.arange(0, contrast_AIF_numpy.size)
    #     # estimated_concentration2 = estimate_concentration([.25, .01], contrast_AIF_numpy, time_interval)

    #     # difference_term = observed_concentration - estimated_concentration
    #     # # print sum(power(difference_term, 2))

    #     # plt.plot(time_series, estimated_concentration, 'r--', time_series, estimated_concentration2, 'g--', time_series, observed_concentration, 'b--')
    #     # plt.show()

    #     delta = .01
    #     x = np.arange(0, .35, delta)
    #     delta = .01
    #     y = np.arange(0, .5, delta)
    #     X, Y = np.meshgrid(x, y)
    #     Z = np.copy(X)

    #     W = x
    #     x1 = np.copy(x)
    #     y1 = np.copy(x)

    #     for k_idx, ktrans in enumerate(x):
    #         for v_idx, ve in enumerate(y):
    #             estimated_concentration = estimate_concentration([ktrans, ve], contrast_AIF_numpy, time_interval)
    #             difference_term = observed_concentration - estimated_concentration
    #             Z[v_idx, k_idx] = sum(power(difference_term, 2))

    #         estimated_concentration = estimate_concentration([ktrans, .1], contrast_AIF_numpy, time_interval)
    #         difference_term = observed_concentration - estimated_concentration
    #         W[k_idx] = sum(power(difference_term, 2))

    #     CS = plt.contourf(X,Y,Z, 30)
    #     plt.clabel(CS, inline=1, fontsize=10)
    #     plt.show()

    #     # plt.plot(optimization_path)
    #     # plt.show()

def estimate_concentration(params, contrast_AIF_numpy, time_interval):

    # Notation is very inexact here. Clean it up later.

    estimated_concentration = [0]
    # if params[0] > 10 or params[1] > 10:
    #   return estimated_concentration

    append = estimated_concentration.append
    e = math.e

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

    # Quick, error prone convolution method
    # print estimated_concentration
        # res = np.exp(-1*kep*time_series)
        # estimated_concentration = ktrans * np.convolve(contrast_AIF_numpy, res) * time_series[1]
        # estimated_concentration = estimated_concentration[0:np.size(res)]

    return estimated_concentration

if __name__ == "__main__":
	pass