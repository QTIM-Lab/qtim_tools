""" This feature set calculates basic matrix-wise statistics. I'm considering
    letting other feature sets import this function, but it might
    add needless complexitly down the road.
"""

from __future__ import division

from ..qtim_utilities import nifti_util

import numpy as np
from scipy import stats

standard_features = ['mean','min','max','median','range','standard_deviation','variance','energy','entropy','kurtosis','skewness','COV']

def calc_mean(image_numpy):
    return np.mean(image_numpy)

def calc_min(image_numpy):
    return np.min(image_numpy)

def calc_max(image_numpy):
    return np.max(image_numpy)

def calc_median(image_numpy):
    return np.median(image_numpy)

def calc_range(image_numpy):
    return np.max(image_numpy) - np.min(image_numpy)

def calc_std(image_numpy):
    return np.std(image_numpy)

def calc_variance(image_numpy):
    return np.var(image_numpy)

def calc_energy(image_numpy):
    return np.sqrt(np.sum(image_numpy ** 2))

def calc_entropy(image_numpy):

    """ Note this silly solution to deal with log(0)
        Surely there must be a better way.
    """

    entropy_image = np.copy(image_numpy)
    entropy_image[entropy_image == 0] = 1
    return np.sum(entropy_image * -1 * (np.log(entropy_image)))

def calc_kurtosis(image_numpy):
    return stats.kurtosis(image_numpy)

def calc_skewness(image_numpy):
    return stats.skew(image_numpy)

def calc_COV(image_numpy):
    return np.std(image_numpy) / np.mean(image_numpy)

def statistics_features(image, features=standard_features, mask_value=0):

    if isinstance(features, basestring):
        features = [features,]

    results = np.zeros(len(features), dtype=float)
    stats_image = np.ravel(image[image != mask_value])
    # nifti_util.check_image(image)

    for f_idx, current_feature in enumerate(features):

        if current_feature == 'mean':
            output = calc_mean(stats_image)
        if current_feature == 'min':
            output = calc_min(stats_image)
        if current_feature == 'max':
            output = calc_max(stats_image)
        if current_feature == 'median':
            output = calc_median(stats_image)
        if current_feature == 'range':
            output = calc_range(stats_image)
        if current_feature == 'standard_deviation':
            output = calc_std(stats_image)
        if current_feature == 'variance':
            output = calc_variance(stats_image)
        if current_feature == 'energy':
            output = calc_energy(stats_image)
        if current_feature == 'entropy':
            output = calc_entropy(stats_image)
        if current_feature == 'kurtosis':
            output = calc_kurtosis(stats_image)
        if current_feature == 'skewness':
            output = calc_skewness(stats_image)
        if current_feature == 'COV':
            output = calc_COV(stats_image)
        results[f_idx] = output

    return results

def featurename_strings(features=standard_features):
    return features

def feature_count(features=standard_features):
    if isinstance(features, basestring):
        features = [features,]
    return len(features)
