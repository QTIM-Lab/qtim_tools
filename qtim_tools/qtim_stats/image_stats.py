import numpy as np
import os

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti
from qtim_tools.test_data.load import load_test_file
from qtim_tools.qtim_utilities.file_util import nifti_splitext, grab_files_recursive

def correlation_analysis(input_volume):

    image_numpy = convert_input_2_numpy(input_volume)

    displacement_list = np.mgrid[1:17:1, 1:17:1, 1:17:1].reshape(3,-1).T - 8

    output_correlation_matrix = np.zeros((17,17,17), dtype=float)

    for displacement in displacement_list:
        print displacement
        x, y, z = displacement
        slice_list = []
        displacement_slice_list = []

        for axis in [x,y,z]:
            if axis < 0:
                slice_list += [slice(-axis, None, 1)]
                displacement_slice_list += [slice(0, axis, 1)]
            elif axis > 0:
                slice_list += [slice(0, -axis, 1)]
                displacement_slice_list += [slice(axis, None, 1)]
            else:
                slice_list += [slice(None)]
                displacement_slice_list += [slice(None)]

        print slice_list
        print displacement_slice_list

        compare_array_1 = image_numpy[slice_list]
        compare_array_2 = image_numpy[displacement_slice_list]

        print compare_array_1.shape
        print compare_array_2.shape

        correlation = np.corrcoef(compare_array_1.reshape(-1), compare_array_2.reshape(-1))
        print correlation
        print '\n'

        output_correlation_matrix[x+8, y+8, z+8] = correlation[1,0]

    save_numpy_2_nifti(output_correlation_matrix, None, nifti_splitext(os.path.basename(input_volume))[0] + '_array' + nifti_splitext(os.path.basename(input_volume))[-1])






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

if __name__ == '__main__':
    # correlation_analysis(load_test_file('sample_mri'))
    correlation_analysis('/qtim2/users/data/FMS/ANALYSIS/COREGISTRATION/FMS_01/VISIT_01/FMS_01-VISIT_01-MPRAGE_POST_r_T2.nii.gz')
    correlation_analysis('/qtim2/users/data/FMS/ANALYSIS/COREGISTRATION/FMS_01/VISIT_01/FMS_01-VISIT_01-SUV_r_T2.nii.gz')
    correlation_analysis('/qtim2/users/data/FMS/ANALYSIS/COREGISTRATION/FMS_01/VISIT_01/FMS_01-VISIT_01-DSC_GE_r_T2.nii.gz')
    correlation_analysis('/qtim2/users/data/FMS/ANALYSIS/COREGISTRATION/FMS_01/VISIT_01/FMS_01-VISIT_01-FLAIR_r_T2.nii.gz')