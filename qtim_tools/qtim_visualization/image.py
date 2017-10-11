""" This module is loosely defined as those visualization operations
    which take in data and generate an image. For example, a mosaic image
    from an inpute medical volume. May be collapsed at a later date into
    another module if use cases are insufficient.
"""

import numpy as np

import os
import glob
from shutil import copy, move
import matplotlib.pyplot as plt

from qtim_tools.qtim_utilities.array_util import generate_label_outlines
from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy

def create_mosaic(input_volume, outfile=None, label_volume=None, generate_outline=True, mask_value=0, step=1, dim=2, cols=8, label_buffer=5, rotate_90=3, flip=True):

    """ This creates a mosaic of 2D images from a 3D Volume.

        Script in progress, much TODO

        Parameters
        ----------

        Returns
        -------

    """

    image_numpy = convert_input_2_numpy(input_volume)
    if step is None:
        step = 1

    if label_volume is not None:

        label_numpy = convert_input_2_numpy(label_volume)

        if generate_outline:
            label_numpy = generate_label_outlines(label_numpy, dim, mask_value)

        # This is fun in a wacky way, but could probably be done more concisely and effeciently.
        mosaic_selections = []
        for i in xrange(label_numpy.shape[dim]):
            label_slice = np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])
            if np.sum(label_slice) != 0:
                mosaic_selections += range(i-label_buffer, i+label_buffer)
        mosaic_selections = np.unique(mosaic_selections)
        mosaic_selections = mosaic_selections[mosaic_selections >= 0]
        mosaic_selections = mosaic_selections[mosaic_selections <= image_numpy.shape[dim]]
        mosaic_selections = mosaic_selections[::step]

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]
        color_range_label = [np.min(label_numpy), np.max(label_numpy)]

        # One day, specify rotations by affine matrix.
        # Is test slice necessary? Operate directly on shape if possible.
        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in xrange(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_image_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)
        mosaic_label_numpy = np.zeros_like(mosaic_image_numpy)
        
        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]]), rotate_90)
            label_slice = np.rot90(np.squeeze(label_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]]), rotate_90)

            # Again, specify from affine matrix if possible.
            if flip:
                image_slice = np.fliplr(image_slice)
                label_slice = np.fliplr(label_slice)

            if image_slice.size > 0:
                mosaic_image_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = image_slice
                mosaic_label_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = label_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        mosaic_label_numpy = np.ma.masked_where(mosaic_label_numpy == 0, mosaic_label_numpy)

        if outfile is not None:
            fig = plt.figure(figsize=(mosaic_image_numpy.shape[0]/100, mosaic_image_numpy.shape[1]/100), dpi=100, frameon=False)
            plt.margins(0,0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')
            plt.imshow(mosaic_label_numpy, 'jet', vmin=color_range_label[0], vmax=color_range_label[1], interpolation='none')
            
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=1000)
            plt.clf()
            plt.close()

    else:

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]

        test_slice = np.rot90(np.squeeze(image_numpy[[slice(None) if k != dim else slice(0, 1) for k in xrange(3)]]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_selections = np.arange(image_numpy.shape[dim])[::step]
        print mosaic_selections
        mosaic_image_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)

        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.squeeze(image_numpy[[slice(None) if k != dim else slice(i, i+1) for k in xrange(3)]])

            image_slice = np.rot90(image_slice, rotate_90)
            
            if flip:
                image_slice = np.fliplr(image_slice)

            mosaic_image_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = image_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        if outfile != '':
            fig = plt.figure(figsize=(mosaic_image_numpy.shape[0]/100, mosaic_image_numpy.shape[1]/100), dpi=100, frameon=False)
            plt.margins(0,0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')

            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=500) 
            plt.clf()
            plt.close()

def run_test():

    return

if __name__ == '__main__':
    run_test()