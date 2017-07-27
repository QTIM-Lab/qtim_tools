""" This module will create python dictionaries linking together image/volume URLs, label URLs, attributes, and data. These databases can then be exported into formats acceptable by d3.js, Shiny, or other visualization apps.
"""

import glob

# from ..core.internals import qtim_Exception

# def create_image_database(input_dictionary={}, input_array=[], image_file_col=0, label_file_col=1):

#     """ Take in either a string array or a directory full of images, and return a python dictionary that is pre-formatted to work
#         well with all other qtim_visualization generators. Can
#         optionally associate images with a list of labels.

#         Parameters
#         ----------
#         image_list : list, optional
#             A list of filenames to enter into the image database. Each
#             filename will serve as a key in the dictionary. Can also
#             take image filename / label filename pairs.
#         image_regex: str, optional
#             A linux-type search expression for image filenames. For 
#             example, to grab all jpg files in a folder, enter 
#             '~/folder/*.jpg'. Can be combined with image_list.
#         label_identifier: str, optional
#             A identification phrase to search for labels from provided
#             image filenames. Filenames containing both an image filename
#             and the label_identifier will be added as labels.
#         ignore_missings: bool, optional
#             If data cannot be verified to exist (e.g. filenames), or
#             label data cannot be found, supply a missing data identifier
#             instead of returning an error.
#     """

#     if len(input_dictionary) > 0:
#         return input_dictionary

#     return

def run_test():
    pass

if __name__ == '__main__':
    run_test()