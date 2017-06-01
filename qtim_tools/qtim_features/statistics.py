""" This feature set calculates basic matrix-wise statistics. I'm considering
    letting other feature sets import this function, but it might
    add needless complexitly down the road.
"""

from __future__ import division

import numpy as np
from scipy import stats

from ..qtim_utilities import nifti_util
from ..qtim_utilities.format_util import convert_input_2_numpy
from ..qtim_preprocessing.threshold import crop_with_mask

standard_bins = [[-np.inf, -1000],[-1000, -950],[-950, -650],[-650, -300],[-300, 0],[0, 100],[100,600],[600, np.inf]]
standard_bin_labels = []
for label in standard_bins:
    standard_bin_labels += ['histogram_percent_' + str(label[0]) + '_' + str(label[1])]

standard_features = ['mean','min','max','median','range','standard_deviation','variance','energy','entropy','kurtosis','skewness','COV'] + standard_bin_labels

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
        Also I am not sure if entropy is valid for negative
        numbers in general..
    """

    entropy_image = np.copy(image_numpy)
    entropy_image[entropy_image <= 0] = 1
    return np.sum(entropy_image * -1 * (np.log(entropy_image)))

def calc_kurtosis(image_numpy):
    return stats.kurtosis(image_numpy, axis=None)

def calc_skewness(image_numpy):
    return stats.skew(image_numpy, axis=None)

def calc_COV(image_numpy):
    return np.std(image_numpy) / np.mean(image_numpy)

def calc_voxel_count(image_numpy, mask_value=0):
    return image_numpy[image_numpy != mask_value].size

def calc_intensity_histogram(image_numpy, bins, mask_value=0):
    
    # This function is not generalizable yet.. TODO.
    voxel_count = calc_voxel_count(image_numpy, mask_value)
    histo_counts = []

    for histo_bin in standard_bins:
        histo_counts += [((histo_bin[0] < image_numpy) & (image_numpy < histo_bin[1])).sum() / voxel_count]

    return histo_counts

statistics_dict = { 'mean': calc_mean,
                    'min': calc_min,
                    'max': calc_max,
                    'median': calc_median,
                    'range': calc_range,
                    'standard_deviation': calc_std,
                    'variance': calc_variance,
                    'energy': calc_energy,
                    'entropy': calc_entropy,
                    'kurtosis': calc_kurtosis,
                    'skewness': calc_skewness,
                    'COV': calc_COV,
                    'histogram_percent': calc_intensity_histogram
                    }

def qtim_statistic(input_data, statistics, label_data='', mask_value=0, return_label=[], additional_parameters=''):

    """ TODO: Documentation. This should replace the existing statistics function for the feature extractor.
    """

    outputs = []
    if isinstance(statistics, basestring):
        statistics = [statistics,]

    input_numpy = convert_input_2_numpy(input_data)

    if label_data != '':
        input_numpy = crop_with_mask(input_numpy, label_data, mask_value=mask_value, return_labels=return_label)
    
    stats_numpy = np.ravel(input_numpy[input_numpy > mask_value])    

    for statistic in statistics:

        if statistics_dict[statistic] == []:
            print 'No statistics by that keyword. Returning blank...'
            outputs += ['']

        try:
            outputs += [statistics_dict[statistic](stats_numpy)]
        except:
            outputs += ['NA']

    return outputs

def statistics_features(image, features=standard_features, label_file='', mask_value=0):

    if isinstance(features, basestring):
        features = [features,]

    results = np.zeros(len(features), dtype=float)
    stats_image = np.ravel(image[image != mask_value])

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

        # This histogram method is very fragile.
        if 'histogram_percent' in current_feature:
            output = calc_intensity_histogram(stats_image, f_idx, mask_value)
            results[f_idx:] = output
            break

        results[f_idx] = output

    return results

def featurename_strings(features=standard_features):
    return features

def feature_count(features=standard_features):
    if isinstance(features, basestring):
        features = [features,]
    return len(features)
