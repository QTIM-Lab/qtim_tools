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

def human_sort():

    """ TODO: make alphanumeric sorting function that also sorts numbers
    """