""" All features as of yet have been copied from the "shape-based
    features" list in the HeterogeneityCad module in Slicer. All of
    these will need to be tested against ground-truth soon.
"""

from __future__ import division

import numpy as np

from collections import namedtuple
from scipy import signal
from scipy.spatial.distance import cdist, pdist
from skimage.measure import find_contours
from skimage.morphology import binary_erosion, disk, convex_hull_image
from math import atan2, degrees

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Point_3D(namedtuple('Point', 'x y z')):

    __slots__ = ()

    @property
    def length(self, **kwargs):
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
    
    def __sub__(self, p, **kwargs):
        return Point_3D(self.x - p.x, self.y - p.y, self.z - p.z)
    
    def __str__(self, **kwargs):
        return 'Point: x=%6.3f  y=%6.3f z=%6.3f length=%6.3f' % (self.x, self.y, self.z, self.length)


class Point_2D(namedtuple('Point', 'x y')):

    __slots__ = ()

    @property
    def length(self, **kwargs):
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __sub__(self, p, **kwargs):
        return Point_2D(self.x - p.x, self.y - p.y)
    
    def __str__(self, **kwargs):
        return 'Point: x=%6.3f  y=%6.3f  length=%6.3f' % (self.x, self.y, self.length)


def convolve_3d(image, kernel, skip_zero=True, **kwargs):

    """ Currently only works for 3x3 kernels.
        Also, as usual, there must be a better
        way to do this.
    """

    convolve_image = np.zeros_like(image)
    padded_image = np.lib.pad(image, (1, ), 'constant')

    for x in xrange(image.shape[0], **kwargs):
        for y in xrange(image.shape[1], **kwargs):
            for z in xrange(image.shape[2], **kwargs):

                if image[x, y, z] == 0 and skip_zero:
                    pass
                else:
                    image_subset = padded_image[(x):(x+3), (y):(y+3), (z):(z+3)]
                    convolve_image[x, y, z] = np.sum(np.multiply(image_subset, kernel))

    return convolve_image


def calc_voxel_count(image, mask_value=0, **kwargs):
    return image[image != mask_value].size


def calc_volume(image, pixdims, mask_value=0, **kwargs):
    return pixdims[0] * pixdims[1] * pixdims[2] * calc_voxel_count(image, mask_value)


def calc_surface_area(image, pixdims, mask_value=0, **kwargs):

    """ Reminder: Verify on real-world data.
        Also, some of the binarization feels clumsy/ineffecient.
        Also, this will over-estimate surface area, because
        it is counting cubes instead of, say, triangular
        surfaces
    """
    
    edges_kernel = np.zeros((3, 3, 3), dtype=float)
    edges_kernel[1, 1, 0] = -1 * pixdims[0] * pixdims[1]
    edges_kernel[0, 1, 1] = -1 * pixdims[1] * pixdims[2]
    edges_kernel[1, 0, 1] = -1 * pixdims[0] * pixdims[2]
    edges_kernel[1, 2, 1] = -1 * pixdims[0] * pixdims[2]
    edges_kernel[2, 1, 1] = -1 * pixdims[1] * pixdims[2]
    edges_kernel[1, 1, 2] = -1 * pixdims[0] * pixdims[1]
    edges_kernel[1, 1, 1] = 1 * (2 * pixdims[0] * pixdims[1] + 2 * pixdims[0] * pixdims[2] + 2 * pixdims[1] * pixdims[2])

    label_numpy = np.copy(image)
    label_numpy[label_numpy != mask_value] = 1
    label_numpy[label_numpy == mask_value] = 0

    edge_image = signal.convolve(label_numpy, edges_kernel, mode='same')
    edge_image[edge_image < 0] = 0

    # nifti_util.check_image(edge_image)

    surface_area = np.sum(edge_image)

    return surface_area


def calc_surface_area_vol_ratio(surface_area, volume, **kwargs):
    return surface_area / volume


def calc_compactness(surface_area, volume, **kwargs):
    return volume / ((np.pi**0.5) * (surface_area**(2 / 3)))


def calc_compactness_alternate(surface_area, volume, **kwargs):
    return 36 * np.pi * volume**2 / surface_area**3


def calc_spherical_disproportion(surface_area, volume, **kwargs):
    return surface_area / (4 * np.pi * (((3 * volume) / (4 * np.pi))**(1 / 3))**2)


def calc_sphericity(surface_area, volume, **kwargs):
    return (np.pi**(1 / 3)) * ((6 * volume)**(2 / 3)) / surface_area


def compute_pairwise_distances(P1, P2, min_length=0, **kwargs):
    
    if len(P1[0]) == 2:
        Point = Point_2D
    else:
        Point = Point_3D

    euc_dist_matrix = cdist(P1, P2, metric='euclidean')
    indices = []
    for x in range(euc_dist_matrix.shape[0]):
        for y in range(euc_dist_matrix.shape[1]):

            p1 = Point(*P1[x])
            p2 = Point(*P1[y])
            d = euc_dist_matrix[x, y]
            
            if p1 == p2 or min_length < 0:
                continue

            indices.append([p1, p2, d])

    return euc_dist_matrix, sorted(indices, key=lambda x: x[2], reverse=True)


def calc_max_2d_distance(image, pixdims, **kwargs):

    total_max = 0

    for z_slice in xrange(image.shape[2]):

        image_slice = image[..., z_slice]

        h, w = np.sum(np.max(image_slice > 0, axis=1)), np.sum(np.max(image_slice > 0, axis=0))

        # Dilate slightly to prevent self-intersections, and compute contours
        dilated = binary_erosion(image_slice, disk(radius=1)).astype('uint8') * 255
        contours = find_contours(dilated, level=1)

        if len(contours) == 0:
            continue

        # Calculate pairwise distances over boundary
        outer_contour = contours[0]  # this assumption should always hold...
        outer_contour[:, 0] *= pixdims[0]
        outer_contour[:, 1] *= pixdims[1]

        euc_dist_matrix, ordered_diameters = compute_pairwise_distances(outer_contour, outer_contour, min_length=w)
        x, y, distance = ordered_diameters[0]

        current_max = distance

        if current_max > total_max:
            total_max = current_max

    return total_max


def split(u, v, points):
    # return points on left side of UV
    return [p for p in points if np.cross(p - u, v - u) < 0]


def extend(u, v, points):
    if not points:
        return []

    # find furthest point W, and split search to WV, UW
    w = min(points, key=lambda p: np.cross(p - u, v - u))
    p1, p2 = split(w, v, points), split(u, w, points)
    return extend(w, v, p1) + [w] + extend(u, w, p2)


def convex_hull(points):
    # find two hull points, U, V, and split to left and right search
    u = min(points, key=lambda p: p[0])
    v = max(points, key=lambda p: p[0])
    left, right = split(u, v, points), split(v, u, points)

    # find convex hull on each side
    return [v] + extend(u, v, left) + [u] + extend(v, u, right) + [v]


def calc_max_3d_distance(image, pixdims, **kwargs):

    total_max = 0
    truth_indices = np.where(image.astype(bool))
    truth_indices = np.asarray(truth_indices).T

    # convex_hull_slice = convex_hull_image(image)
    # verts, faces, normals, values = marching_cubes_lewiner(image)

    convex_hull_vertices = []

    for z_slice in xrange(image.shape[2]):

        slice_indices = np.asarray(np.where(image[..., z_slice])).T  # Won't work for non-binary labels.
        
        if len(slice_indices) == 0:
            continue

        convex_hull_slice = convex_hull(slice_indices)
        convex_hull_slice = [np.array([x[0], x[1], z_slice]) for x in convex_hull_slice]
        convex_hull_vertices += convex_hull_slice

    convex_hull_vertices = np.array(convex_hull_vertices).astype(float)
    
    convex_hull_vertices[:, 0] *= pixdims[0]
    convex_hull_vertices[:, 1] *= pixdims[1]
    convex_hull_vertices[:, 2] *= pixdims[2]

    euc_dist_matrix, ordered_diameters = compute_pairwise_distances(convex_hull_vertices, convex_hull_vertices, metric='euclidean')

    x, y, distance = ordered_diameters[0]

    total_max = distance

    return total_max


_default_features = {'voxel_count': calc_voxel_count,
                        'volume': calc_volume,
                        'surface_area': calc_surface_area,
                        'volume_surface_area_ratio': calc_surface_area_vol_ratio,
                        'compactness': calc_compactness,
                        'compactness_alternate': calc_compactness_alternate,
                        'spherical_disproportion': calc_spherical_disproportion,
                        'sphericity': calc_sphericity,
                        '2d_max_distance': calc_max_2d_distance,
                        '3d_max_distance': calc_max_3d_distance}
_default_feature_names = _default_features.keys()


def morphology_features(image, attributes, features=['voxel_count', 'volume', 'surface_area', 'volume_surface_area_ratio', 'compactness', 'compactness_alternate', 'spherical_disproportion', 'sphericity', '2d_max_distance', '3d_max_distance'], mask_value=0, **kwargs):

    if isinstance(features, basestring, **kwargs):
        features = [features, ]

    results = np.zeros(len(features), dtype=float)
    pixdims = attributes['pixdim'][1:4]

    volume = calc_volume(image, pixdims, mask_value)
    surface_area = calc_surface_area(image, pixdims, mask_value)

    for f_idx, current_feature in enumerate(features, **kwargs):

        output = _default_features[current_feature](image=image, surface_area=surface_area, volume=volume, pixdims=pixdims, mask_value=mask_value)

        results[f_idx] = output

    return results


def featurename_strings(features=['voxel_count', 'volume', 'surface_area', 'volume_surface_area_ratio', 'compactness', 'compactness_alternate', 'spherical_disproportion', 'sphericity', '2d_max_distance', '3d_max_distance'], **kwargs):
    return features


def feature_count(features=['voxel_count', 'volume', 'surface_area', 'volume_surface_area_ratio', 'compactness', 'compactness_alternate', 'spherical_disproportion', 'sphericity', '2d_max_distance', '3d_max_distance'], **kwargs):
    if isinstance(features, basestring, **kwargs):
        features = [features, ]
    return len(features)