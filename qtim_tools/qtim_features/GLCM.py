""" This is the beginning of a program meant to calculate GLCM features on 3D
    arrays generated from NIFTI images. It is based off of the greycomatrix,
    greycoprops and other GLCM utility code lifted straight from scikit-image.
    Goals for this program: generalize sci-kit's 2D GLCM to 3D, optimize 3D
    GLCM calculation, figure out a way to aggregate 2D calculations into a 
    useful measure within 3D images, and make a GLCM properties filtered image
    from smaller GLCM calculations in subregions of the original image. No matter
    what, this function could use a lot of cleaning.
"""

from __future__ import division

from qtim_tools.qtim_utilities.nifti_util import assert_nD

from math import sin, cos
import numpy as np

def glcm_2d_aggregate(image, distances, angles, levels=None, symmetric=False, normed=True, aggregate_axis=2, method="sum", masked=True, mask_value=0, test=False):
    
    # GLCM currently fails for "1-d" stacks of pixels, and for non-axial slices unless aggreggate_axis is set to -1.
    # Will have to modify assertions checking dimensions to accomodate these situations.

    if test:
        image = np.zeros((20,20))
        for x in xrange(1,11):
            for y in xrange(1,11):
                image[x,y] = x*y
        return glcm_2d(image, distances, angles, levels, symmetric, normed=False, mask_value=mask_value)

    assert_nD(image, 3)

    image = np.ascontiguousarray(image)

    image_max = image.max()

    if np.issubdtype(image.dtype, np.float):
        raise ValueError("Float images are not supported by greycomatrix. "
                         "Convert the image to an unsigned integer type.")

    if levels is None:
        levels = 256

    # for image type > 8bit, levels must be set.
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    if image_max >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

    if aggregate_axis == -1:
        aggregate_axis = np.argmin(image.shape)
        print '2-D GLCM aggregation axis chosen automatically at ' + str(aggregate_axis)

    nSlice = image.shape[aggregate_axis]
    result_GLCM = np.zeros((levels, levels, len(distances), len(angles)),
                 dtype=np.uint32, order='C')
    if normed:
        result_GLCM = result_GLCM.astype(float)

    if method == "maximal_slice":

        image_slice = np.squeeze(image[[slice(None) if k != aggregate_axis else slice(0, 1) for k in xrange(3)]])
        maximal = [0, np.zeros_like(image_slice)]

        for i in xrange(nSlice):
            
            # Full disclosure: I'm not entirely sure how this works, 
            # but this code slices and image by an arbitrary axis
            image_slice = np.squeeze(image[[slice(None) if k != aggregate_axis else slice(i, i+1) for k in xrange(3)]])

            test_maximal = (image_slice != mask_value).sum()

            if test_maximal >= maximal[0]:
                maximal[0] = test_maximal
                maximal[1] = image_slice

        result_GLCM = glcm_2d(maximal[1], distances, angles, levels, symmetric, normed, mask_value)
        return result_GLCM

    elif method == "sum" or method == "average":

        for i in xrange(nSlice):
            
            # Full disclosure: I'm not entirely sure how this works, 
            # but this code slices an image by an arbitrary axis
            image_slice = np.squeeze(image[[slice(None) if k != aggregate_axis else slice(i, i+1) for k in xrange(3)]])
            slice_GLCM = glcm_2d(image_slice, distances, angles, levels, symmetric, normed=False, mask_value=mask_value)
            if method == "sum":
                result_GLCM += slice_GLCM
            elif method == "average":
                size = i + 1
                result_GLCM = result_GLCM + ((1 / (size + 1)) * (slice_GLCM - result_GLCM))

        if normed:
            result_GLCM = result_GLCM.astype(np.float64)
            glcm_sums = np.apply_over_axes(np.sum, result_GLCM, axes=(0, 1))
            glcm_sums[glcm_sums == 0] = 1
            result_GLCM /= glcm_sums
            
        return result_GLCM

    else:
        raise ValueError("You have chosen an invalid aggregation method. Accepted methods are \'sum\', \'average\', and \'maximal_slice.\'")


def glcm_2d(image, distances, angles, levels=None, symmetric=False,
                 normed=True, mask_value=0):
    """Calculate the grey-level co-occurrence matrix.
    A grey level co-occurrence matrix is a histogram of co-occurring
    greyscale values at a given offset over an image.
    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.
    Returns
    -------
    P : 4-D ndarray
        The grey-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that grey-level `j`
        occurs at a distance `d` and at an angle `theta` from
        grey-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.
    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    .. [2] Pattern Recognition Engineering, Morton Nadler & Eric P.
           Smith
    .. [3] Wikipedia, http://en.wikipedia.org/wiki/Co-occurrence_matrix
    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)
    """
    assert_nD(image, 2)
    assert_nD(distances, 1, 'distances')
    assert_nD(angles, 1, 'angles')

    image = np.ascontiguousarray(image)

    image_max = image.max()

    if np.issubdtype(image.dtype, np.float):
        raise ValueError("Float images are not supported by greycomatrix. "
                         "Convert the image to an unsigned integer type.")

    # for image type > 8bit, levels must be set.
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    if levels is None:
        levels = 256

    if image_max >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)

    P = np.zeros((levels, levels, len(distances), len(angles)),
                 dtype=np.uint32, order='C')

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, mask_value)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLMC
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P

def _glcm_loop(image, distances, angles, levels, out, mask_value):
    """Perform co-occurrence matrix accumulation.
    Parameters
    ----------
    image : ndarray
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : ndarray
        List of pixel pair distance offsets.
    angles : ndarray
        List of pixel pair angles in radians.
    levels : int
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image).
    out : ndarray
        On input a 4D array of zeros, and on output it contains
        the results of the GLCM computation.
    """

    """
    Interesting note about calculating contrast at different distances. Contrast calculated at large
    distances -- say, 10 pixels vs 1, will not count pixels that don't have neighbors 10 pixels apart
    this will artifically deflate the number of hits in those situations. Unknown how to fix, or even
    what fixing means in this situation.
    """

    # with nogil:
    rows = image.shape[0]
    cols = image.shape[1]
    configurations = [distances] + [angles]

    for d_idx in range(distances.shape[0]):
        distance = distances[d_idx]
        for a_idx in range(angles.shape[0]):
            angle = angles[a_idx]
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]
                    if i == mask_value:
                        continue
                    # compute the location of the offset pixel
                    row = r + int(round(sin(angle) * distance))
                    col = c + int(round(cos(angle) * distance))

                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and col >= 0 and col < cols:
                        j = image[row, col]

                        # make sure values are within the level parameters, and that offset is not in the mask
                        if i >= 0 and i < levels and j >= 0 and j < levels and j != mask_value:
                        # if i >= 0 and i < levels and j >= 0 and j < levels:                            
                            out[i, j, d_idx, a_idx] += 1

def glcm_features_calc(P, props=['contrast', 'dissimilarity', 'homogeneity', 'ASM','energy','correlation'], distances=None, angles=None, out='list'):
    """Calculate texture properties of a GLCM.
    Compute a feature of a grey level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:
    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]
    Parameters
    ----------
    P : ndarray
        Input array. `P` is the grey-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that grey-level j
        occurs at a distance d and at an angle theta from
        grey-level i.
    prop : ['contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'], optional
        A string array of properties for the GLCM to compute. The default is all properties.
    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.
    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = greycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = greycoprops(g, 'contrast')
    >>> contrast
    array([[ 0.58333333,  1.        ],
           [ 1.25      ,  2.75      ]])
    """
    (num_level, num_level2, num_dist, num_angle) = P.shape
    assert num_level == num_level2
    assert num_dist > 0
    assert num_angle > 0

    if isinstance(props, basestring):
        props = [props,]
    num_props = len(props)

    results = np.zeros((num_dist, num_angle, num_props), dtype=float)

    for p_idx, current_prop in enumerate(props):

        # create weights for specified current_property
        I, J = np.ogrid[0:num_level, 0:num_level]
        if current_prop == 'contrast':
            weights = (I - J) ** 2
        elif current_prop == 'dissimilarity':
            weights = np.abs(I - J)
        elif current_prop == 'homogeneity':
            weights = 1. / (1. + (I - J) ** 2)
        elif current_prop in ['ASM', 'energy', 'correlation']:
            pass
        else:
            raise ValueError('%s is an invalid property' % (current_prop))

        # compute current_property for each GLCM
        # Note that the defintion for "Energy" varies between studies.
        if current_prop == 'energy':
            asm = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
            results[:,:,p_idx] = np.sqrt(asm)
        elif current_prop == 'ASM':
            results[:,:,p_idx] = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
        elif current_prop == 'correlation':
            tempresults = np.zeros((num_dist, num_angle), dtype=np.float64)
            I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
            J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
            diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
            diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

            std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                               axes=(0, 1))[0, 0])
            std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                               axes=(0, 1))[0, 0])
            cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                     axes=(0, 1))[0, 0]

            # handle the special case of standard deviations near zero
            mask_0 = std_i < 1e-15
            mask_0[std_j < 1e-15] = True
            tempresults[mask_0] = 1

            # handle the standard case
            mask_1 = mask_0 == False
            tempresults[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
            results[:,:,p_idx] = tempresults
        elif current_prop in ['contrast', 'dissimilarity', 'homogeneity']:
            if current_prop == 'contrast':
                pass
            weights = weights.reshape((num_level, num_level, 1, 1))
            results[:,:,p_idx] = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    if out == 'list':
        results = results.reshape(len(props)*num_dist*num_angle)
        return results
    elif out == 'array':
        return results

def glcm_features(image, distances=[1,2,3,4,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], props=['contrast', 'dissimilarity', 'homogeneity', 'ASM','energy','correlation'], levels=None, symmetric=False, normed=True, aggregate_axis=2, method="sum", masked=True, mask_value=0, out='list', return_level_array=False):
    glcm_array = glcm_2d_aggregate(image, distances, angles, levels, symmetric, normed, aggregate_axis, method, masked, mask_value)
    glcm_feats = glcm_features_calc(glcm_array, props, distances, angles, out)
    if return_level_array:
        return [glcm_feats, glcm_array]
    else:
        return glcm_feats

def feature_count(distances=[1,2,3,4,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], props=['contrast', 'dissimilarity', 'homogeneity', 'ASM','energy','correlation']):
    if isinstance(props, basestring):
        props = [props,]
    return len(distances) * len(angles) * len(props)

def featurename_strings(distances=[1,2,3,4,5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], props=['contrast', 'dissimilarity', 'homogeneity', 'ASM','energy','correlation']):
    featurename_list = np.zeros((len(props) * len(distances) * len(angles)), dtype=object)
    featurename_id = 0
    for d_idx in distances:
        for a_idx in angles:
            for p_idx in props:
                featurename_list[featurename_id] = '_'.join(['GLCM', str(d_idx), str(a_idx), p_idx])
                featurename_id += 1
    return featurename_list

if __name__ == "__main__":
    test_array = np.random.randint(0,5,(25,25,25))
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    print glcm_features_calc(glcm_2d_aggregate(test_array, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=5, symmetric=True, method='maximal_slice'), out='list', distances=[1,2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])