import numpy as np

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

def correlation_analysis(input_volume):

    image_numpy = convert_input_2_numpy(input_volume)


def dice_coeffecient(input_label_1, input_label_2):

    """ Computes the Dice coefficient, a measure of set similarity.
        Implementation from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137.

        TODO: Multi-label DICE, weighted dice.

        Parameters
        ----------
        input_label_1 : array-like, bool
            Any array of arbitrary size. If not boolean, will be converted.
        input_label_2 : array-like, bool
            Any other array of identical size. If not boolean, will be converted.
        
        Returns
        -------
        dice : float
            Dice coefficient as a float on range [0,1].
            Maximum similarity = 1
            No similarity = 0
            Both are empty (sum eq to zero) = empty_score

    """

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

    return

def intensity_range(input_volume, percentiles=[.25,.75]):

    """Retrieves a min and max of intensities at two specified percentiles on the intensity histogram.
    This could be useful for thresholding, normalizing, or other tasks. Likely redundant with existing
    numpy feature, TODO: remove
    
    Parameters
    ----------
    input_volume : filename or numpy array
        Input data. If not in an array, will attempt to convert to array.
    percentiles : list, optional
        Histogram percentiles to return. Default is [.25, .75]
    
    Returns
    -------
    list
        A two-item list of intensities at the given percentiles.
    """

    image_numpy = convert_input_2_numpy(input_volume)

    intensity_range = [np.percentile(image_numpy, percentiles[0], interpolation="nearest"), np.percentile(image_numpy, percentiles[1], interpolation="nearest")]

    return intensity_range