""" All features as of yet have been copied from the "shape-based
    features" list in the HeterogeneityCad module in Slicer. All of
    these will need to be tested against ground-truth soon.
"""

from __future__ import division

from ..qtim_utilities import nifti_util

import numpy as np
from scipy import signal

def convolve_3d(image, kernel, skip_zero=True):

    """ Currently only works for 3x3 kernels.
        Also, as usual, there must be a better
        way to do this.
    """

    convolve_image = np.zeros_like(image)
    padded_image = np.lib.pad(image, (1,), 'constant')

    for x in xrange(image.shape[0]):
        for y in xrange(image.shape[1]):
            for z in xrange(image.shape[2]):

                if image[x,y,z] == 0 and skip_zero:
                    pass
                else:
                    image_subset = padded_image[(x):(x+3),(y):(y+3),(z):(z+3)]
                    convolve_image[x,y,z] = np.sum(np.multiply(image_subset,kernel))

    return convolve_image

def calc_voxel_count(image_numpy, mask_value=0):
    return image_numpy[image_numpy != mask_value].size

def calc_volume(image_numpy, pixdims, mask_value=0):
    return pixdims[0] * pixdims[1] * pixdims[2] * calc_voxel_count(image_numpy, mask_value)

def calc_surface_area(image_numpy, pixdims, mask_value=0):

    """ Reminder: Verify on real-world data.
        Also, some of the binarization feels clumsy/ineffecient.
        Also, this will over-estimate surface area, because
        it is counting cubes instead of, say, triangular
        surfaces
    """
    
    edges_kernel = np.zeros((3,3,3),dtype=float)
    edges_kernel[1,1,0] = -1*pixdims[0]*pixdims[1]
    edges_kernel[0,1,1] = -1*pixdims[1]*pixdims[2]
    edges_kernel[1,0,1] = -1*pixdims[0]*pixdims[2]
    edges_kernel[1,2,1] = -1*pixdims[0]*pixdims[2]
    edges_kernel[2,1,1] = -1*pixdims[1]*pixdims[2]
    edges_kernel[1,1,2] = -1*pixdims[0]*pixdims[1]
    edges_kernel[1,1,1] = 1 * (2*pixdims[0]*pixdims[1] + 2*pixdims[0]*pixdims[2] + 2*pixdims[1]*pixdims[2])

    label_numpy = np.copy(image_numpy)
    label_numpy[label_numpy != mask_value] = 1
    label_numpy[label_numpy == mask_value] = 0

    edge_image = signal.convolve(label_numpy, edges_kernel,mode='same')
    edge_image[edge_image < 0] = 0

    # nifti_util.check_image(edge_image)

    surface_area = np.sum(edge_image)

    return surface_area

def surface_area_vol_ratio(surface_area, volume):
    return surface_area / volume

def compactness(surface_area, volume):
    return volume / ((np.pi**0.5)*(surface_area**(2/3)))

def compactness_alternate(surface_area, volume):
    return 36 * np.pi * volume**2 / surface_area**3

def spherical_disproportion(surface_area, volume):
    return surface_area / (4 * np.pi * (((3*volume) / (4*np.pi))**(1/3))**2)

def sphericity(surface_area, volume):
    return (np.pi**(1/3)) * ((6 * volume)**(2/3)) / surface_area

def morphology_features(image, attributes, features=['voxel_count','volume','surface_area','volume_surface_area_ratio','compactness','compactness_alternate','spherical_disproportion','sphericity'], mask_value=0):

    if isinstance(features, basestring):
        features = [features,]

    results = np.zeros(len(features), dtype=float)
    pixdims = attributes['pixdim'][1:4]

    volume = calc_volume(image, pixdims, mask_value)
    surface_area = calc_surface_area(image, pixdims, mask_value)

    for f_idx, current_feature in enumerate(features):

        if current_feature == 'voxel_count':
            output = calc_voxel_count(image, mask_value)
        if current_feature == 'volume':
            output = volume
        if current_feature == 'surface_area':
            output = surface_area
        if current_feature == 'volume_surface_area_ratio':
            output = surface_area_vol_ratio(surface_area, volume)
        if current_feature == 'compactness':
            output = compactness(surface_area, volume)
        if current_feature == 'compactness_alternate':
            output = compactness_alternate(surface_area, volume)
        if current_feature == 'spherical_disproportion':
            output = spherical_disproportion(surface_area, volume)
        if current_feature == 'sphericity':
            output = sphericity(surface_area, volume)

        results[f_idx] = output

    return results

def featurename_strings(features=['voxel_count','volume','surface_area','volume_surface_area_ratio','compactness','compactness_alternate','spherical_disproportion','sphericity']):
    return features

def feature_count(features=['voxel_count','volume','surface_area','volume_surface_area_ratio','compactness','compactness_alternate','spherical_disproportion','sphericity']):
    if isinstance(features, basestring):
        features = [features,]
    return len(features)