""" Various string and file utilties will be contained
    in this module.
"""

import os
import glob

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

def grab_linked_file(input_filename, prefix="", suffix="", regex="", search_folder='', recursive=False, input_format='', output_format=''):
    return

def human_sort(l):

    """ Stolen from Stack Exchange. Sorts alphabetically, but also numerically. How?
        who knows.. Maybe it doesn't even work. TODO: test it.
    """

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
    l.sort( key=alphanum )
    return l