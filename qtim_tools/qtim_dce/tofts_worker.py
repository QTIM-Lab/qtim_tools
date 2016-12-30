# from __future__ import division

# from ..qtim_utilities import nifti_util

# import numpy as np
# import nibabel as nib
# import scipy.optimize
# import matplotlib.pyplot as plt
# import math
# import random
# import os
# import dce_util
# import cProfile
# import re

# import time

# class timewith():
#     def __init__(self, name=''):
#         self.name = name
#         self.start = time.time()

#     @property
#     def elapsed(self):
#         return time.time() - self.start

#     def checkpoint(self, name=''):
#         print '{timer} {checkpoint} took {elapsed} seconds'.format(
#             timer=self.name,
#             checkpoint=name,
#             elapsed=self.elapsed,
#         ).strip()

#     def __enter__(self):
#         return self

#     def __exit__(self, type, value, traceback):
#         self.checkpoint('finished')
#         pass

def simplex_optimize_loop(contrast_image_numpy, contrast_AIF_numpy, time_interval_seconds, mask_value=0, mask_threshold=0, intial_fitting_function_parameters=[1,1]):

	print 'started!'

	np.set_printoptions(threshold=np.nan)
	power = np.power
	sum = np.sum
	e = math.e

	# inexplicable minute conversion; investigate. It looks like the model is just set up that way.
	time_series = np.arange(0, contrast_AIF_numpy.size) / (60 / time_interval_seconds)
	time_interval = time_series[1]

	intial_fitting_function_parameters = [.3, .3]

	def cost_function(params):

		# The estimate concentration function is repeated locally to eke out every last bit of efficiency
		# from this massively looping prgoram.

		# estimated_concentration = estimate_concentration(params, contrast_AIF_numpy, time_interval)

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

		# Notation is very inexact here. Clean it up later.
		for i in xrange(1, np.size(contrast_AIF_numpy)):
			term_A = contrast_AIF_numpy[i] * block_A
			term_B = contrast_AIF_numpy[i-1] * block_B
			append(estimated_concentration[-1]*capital_E + block_ktrans * (term_A - term_B))

		difference_term = observed_concentration - estimated_concentration
		difference_term = power(difference_term, 2)

		return sum(difference_term)

	result_grid = np.zeros((6,5,2),dtype=float)
	output_image = np.zeros((contrast_image_numpy.shape[0:-1] + (3,)), dtype=float)

	if len(contrast_image_numpy.shape) == 3:
		for x in xrange(contrast_image_numpy.shape[0]):
			for y in xrange(contrast_image_numpy.shape[1]):
		# for x_idx, x in enumerate(range(0,50,1)):
			# for y_idx, y in enumerate(range(10,70,1)):

				# Need to think about how to implement masking. Probably np.ma involved. Will likely require
				# editing other np.math functions down the line.

				# if contrast_image_numpy[x,y,0] == mask_value or contrast_image_numpy[x,y,0] < mask_threshold:
					# continue

				print[x,y]

				observed_concentration = contrast_image_numpy[x,y,:]
				
				# with timewith('concentration estimator') as timer:
				result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=0, ftol=.001, xtol=1e-4)

				# This weird parameter transform is a holdover from Xiao's program. I wonder what purpose it serves..
				# ktrans = np.exp(result_params[0]) #ktrans
				# ve = 1 / (1 + np.exp(-result_params[1])) #ve

				ktrans = result_params[0]
				ve = result_params[1]
				# auc = np.trapz(observed_concentration, dx=time_interval_seconds) / np.trapz(contrast_AIF_numpy, dx=time_interval_seconds)


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

		output_image[output_image[:,:,0] > .7] = -.01
		output_image[output_image[:,:,1] > .9] = -.01
		output_image[output_image[:,:,1] < 1e-4] = -.01

		validation_image = np.copy(output_image)
		validation_image = np.ma.masked_equal(validation_image, -.01)
		for y in xrange(0,50,10):
			for x in xrange(10,70,10):
				result_grid[(x)/10 - 1,(y)/10,0] = np.mean((validation_image[x:x+10,y:y+10,0]).flatten())
				result_grid[(x)/10 - 1,(y)/10,1] = np.mean((validation_image[x:x+10,y:y+10,1]).flatten())

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

					result_params = scipy.optimize.fmin(cost_function, intial_fitting_function_parameters, disp=0, ftol=.001, xtol=1e-4)

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

if __name__ == "__main__":
	pass