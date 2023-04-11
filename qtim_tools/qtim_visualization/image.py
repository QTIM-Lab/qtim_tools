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

def create_mosaic(input_volume, outfile=None, label_volume=None, generate_outline=True, mask_value=0, step=1, dim=2, cols=8, label_buffer=5, rotate_90=3, flip=True, dpi=100):

    """This creates a mosaic of 2D images from a 3D Volume.
    
    Parameters
    ----------
    input_volume : TYPE
        Any neuroimaging file with a filetype supported by qtim_tools, or existing numpy array.
    outfile : None, optional
        Where to save your output, in a filetype supported by matplotlib (e.g. .png). If 
    label_volume : None, optional
        Whether to create your mosaic with an attached label filepath / numpy array. Will not perform volume transforms from header (yet)
    generate_outline : bool, optional
        If True, will generate outlines for label_volumes, instead of filled-in areas. Default is True.
    mask_value : int, optional
        Background value for label volumes. Default is 0.
    step : int, optional
        Will generate an image for every [step] slice. Default is 1.
    dim : int, optional
        Mosaic images will be sliced along this dimension. Default is 2, which often corresponds to axial.
    cols : int, optional
        How many columns in your output mosaic. Rows will be determined automatically. Default is 8.
    label_buffer : int, optional
        Images more than [label_buffer] slices away from a slice containing a label pixel will note be included. Default is 5.
    rotate_90 : int, optional
        If the output mosaic is incorrectly rotated, you may rotate clockwise [rotate_90] times. Default is 3.
    flip : bool, optional
        If the output is incorrectly flipped, you may set to True to flip the data. Default is True.
    
    No Longer Returned
    ------------------
    
    Returns
    -------
    output_array: N+1 or N-dimensional array
        The generated mosaic array.
    
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
        for i in range(label_numpy.shape[dim]):
            label_slice = np.squeeze(label_numpy[tuple(slice(None) if k != dim else slice(i, i+1) for k in range(3))])
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
        test_slice = np.rot90(np.squeeze(image_numpy[tuple(slice(None) if k != dim else slice(0, 1) for k in range(3))]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_image_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)
        mosaic_label_numpy = np.zeros_like(mosaic_image_numpy)
        
        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.rot90(np.squeeze(image_numpy[tuple(slice(None) if k != dim else slice(i, i+1) for k in range(3))]), rotate_90)
            label_slice = np.rot90(np.squeeze(label_numpy[tuple(slice(None) if k != dim else slice(i, i+1) for k in range(3))]), rotate_90)

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

        return mosaic_image_numpy

    else:

        color_range_image = [np.min(image_numpy), np.max(image_numpy)]

        test_slice = np.rot90(np.squeeze(image_numpy[tuple(slice(None) if k != dim else slice(0,1) for k in range(3))]), rotate_90)
        slice_width = test_slice.shape[1]
        slice_height = test_slice.shape[0]

        mosaic_selections = np.arange(image_numpy.shape[dim])[::step]
        mosaic_image_numpy = np.zeros((int(slice_height*np.ceil(float(len(mosaic_selections))/float(cols))), int(test_slice.shape[1]*cols)), dtype=float)

        row_index = 0
        col_index = 0

        for i in mosaic_selections:
            image_slice = np.squeeze(image_numpy[tuple(slice(None) if k != dim else slice(i, i+1) for k in range(3))])

            image_slice = np.rot90(image_slice, rotate_90)
            
            if flip:
                image_slice = np.fliplr(image_slice)

            mosaic_image_numpy[int(row_index):int(row_index+slice_height), int(col_index):int(col_index+slice_width)] = image_slice

            if col_index == mosaic_image_numpy.shape[1] - slice_width:
                col_index = 0
                row_index += slice_height 
            else:
                col_index += slice_width

        if outfile is not None:
            fig = plt.figure(figsize=(mosaic_image_numpy.shape[0]/100, mosaic_image_numpy.shape[1]/100), dpi=100, frameon=False)
            plt.margins(0,0)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.imshow(mosaic_image_numpy, 'gray', vmin=color_range_image[0], vmax=color_range_image[1], interpolation='none')

            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=dpi) 
            plt.clf()
            plt.close()

        return mosaic_image_numpy

def run_test():

    return

if __name__ == '__main__':
    run_test()
