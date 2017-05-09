""" Various string and file utilties will be contained
    in this module.
"""

import os
import glob
import fnmatch

def copy_files(infolder, outfolder, name, duplicate=True):

    """ I'm not sure how many of these file-moving helper functions should be
        included. I know I like to have them, but it may be better to have 
        people code their own. It's hard to customize such functions exactly
        to users' needs.
    """

    path = os.path.join(infolder, name)
    print path
    files = glob.glob(path)
    if files == []:
        print 'No files moved. Might you have made an error with the filenames?'
    else:
        for file in files:
            if duplicate:
                copy(file, outfolder)
            else:
                move(file, outfolder)

def grab_files_recursive(input_directory, regex='*'):

    """ A convenience wrapper around os.walk. Returns a list of all files in
        a directory and all of its subdirectories.
    """

    output_list = []

    for root, subFolders, files in os.walk(input_directory):
        if fnmatch.fnmatch(files, regex):
            output_list += [os.path.join(root, files)]

def grab_linked_file(input_filename, prefix="", suffix="", regex="", search_folder='', recursive=False, input_format='', output_format=''):

    pass

    return

def human_sort(l):

    """ Stolen from Stack Exchange. Sorts alphabetically, but also numerically. How?
        who knows.. Maybe it doesn't even work. TODO: test it.
    """

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
    l.sort( key=alphanum )
    return l