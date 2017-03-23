""" This module is a series of scripts relating to extracting signal from noise
	in medical imaging files. Blurring, PCA, other scripts (most likely using
	scikit) will be put here.
"""

import scipy.ndimage
import numpy as np
import random

def gaussian_blur(image_numpy=[], gaussian_blur=0, gaussian_blur_axis=-1, blur_axes=[0]):

    """ Gaussian blurring greatly improves fitting performance on DCE
        phantom data.
    """

    if gaussian_blur > 0:

        output_numpy = np.copy(image_numpy)
        dims = len(image_numpy.shape)

        if blur_axes != [0]:
        	blur_axes = [gaussian_blur]*(dims)

        if gaussian_blur_axis > 0:
            blur_axes[gaussian_blur_axis] = 0 

        output_numpy = scipy.ndimage.filters.gaussian_filter(image_numpy, blur_axes)     

        return output_numpy

    else:
        return image_numpy

def create_PCA_maps(image_numpy, PCA_levels = 10, PCA_axis = -1, return_eigenvalues=False):

	""" Takes in an image with pixels/voxels that change over time, or some other final axis,
		and return an image with the specified amount of PCA levels, ordered from most important
		to least important.
	"""

	# Support for other PCA axes other than the last one? Seems extravagant. Maybe one day..

	pca_curve_data = np.reshape(image_numpy, (np.product(image_numpy.shape[0:-1]), image_numpy.shape[-1]))	

	pca_curve_data_dot_matrix = np.dot(pca_curve_data.T, pca_curve_data)

	# TODO: Understand PCA better, give these more descriptive names.
	eigenvector, eigenmatrix = np.linalg.eig(pca_curve_data_dot_matrix)
	sorting_indices = np.argsort(eigenvector)
	eigenvector_sorted = np.sort(eigenvector)

	eigenmatrix_sorted = eigenmatrix[sorting_indices.astype(int),:]

	pca_curve_data = np.dot(pca_curve_data, eigenmatrix_sorted)

	pca_curve_data = np.reshape(pca_curve_data[..., 0:PCA_levels], image_numpy.shape[0:-1] + (PCA_levels,))

	if return_eigenvalues:
		return pca_curve_data, eigenvector_sorted, eigenmatrix_sorted
	else:
		return pca_curve_data


def PCA_reduce(image_numpy, PCA_levels = 10, PCA_axis = -1):

	pca_maps, eigenvector_sorted, eigenmatrix_sorted = create_PCA_maps(image_numpy, PCA_levels, PCA_axis, return_eigenvalues=True)

	pca_reduced_data = np.reshape(pca_maps, (np.product(pca_maps.shape[0:-1]), pca_maps.shape[-1]))

	pca_reduced_data = np.dot(pca_reduced_data, eigenmatrix_sorted[0:PCA_levels,:])

	pca_reduced_data = np.reshape(pca_reduced_data, image_numpy.shape)

	return pca_reduced_data

if __name__ == '__main__':
	np.random.seed(1)
	print PCA_reduce(np.random.rand(10,10,10,20), PCA_levels = 10).shape
	pass