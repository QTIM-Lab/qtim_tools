""" Various string and file utilties will be contained
    in this module.
"""

import os
import glob
import fnmatch

def copy_files(infolder, outfolder, pattern, duplicate=True):

    """ I'm not sure how many of these file-moving helper functions should be
        included. I know I like to have them, but it may be better to have 
        people code their own. It's hard to customize such functions exactly
        to users' needs.
    """

    path = os.path.join(infolder, pattern)
    files = glob.glob(path)
    if files == []:
        print('No files moved. Might you have made an error with the filenames?')
    else:
        for file in files:
            if duplicate:
                copy(file, outfolder)
            else:
                move(file, outfolder)

def grab_files_recursive(input_directory, regex='*'):

    """ Returns all files recursively in a directory. Essentially a convenience wrapper 
        around os.walk.

        Parameters
        ----------

        input_directory: str
            The folder to search.
        regex: str
            A linux-style pattern to match.

        Returns
        -------
        output_list: list
            A list of found files.
    """

    output_list = []

    for root, subFolders, files in os.walk(input_directory):
        for file in files:
            if fnmatch.fnmatch(file, regex):
                output_list += [os.path.join(root, file)]

    return output_list

def grab_folders_recursive(input_directory, regex='*'):

    """ Returns all folders recursively in a directory. Essentially a convenience wrapper 
        around os.walk.

        Parameters
        ----------

        input_directory: str
            The folder to search.
        regex: str
            A linux-style pattern to match.

        Returns
        -------
        output_list: list
            A list of found files.
    """

    output_list = []

    for root, subFolders, files in os.walk(input_directory):
        for subFolder in subFolders:
            if fnmatch.fnmatch(subFolder, regex):
                output_list += [os.path.join(root, subFolder)]

    return output_list

def nifti_splitext(input_filepath):

    """ os.path.splitext splits a filename into the part before the LAST
        period and the part after the LAST period. This will screw one up
        if working with, say, .nii.gz files, which should be split at the
        FIRST period. This function performs an alternate version of splitext
        which does just that.

        TODO: Make work if someone includes a period in a folder name (ugh).

        Parameters
        ----------
        input_filepath: str
            The filepath to split.

        Returns
        -------
        split_filepath: list of str
            A two-item list, split at the first period in the filepath.

    """

    split_filepath = str.split(input_filepath, '.')

    if len(split_filepath) <= 1:
        return split_filepath
    else:
        return [split_filepath[0], '.' + '.'.join(split_filepath[1:])]

def grab_linked_file(input_filename, prefix="", suffix="", includes="", regex="", linux_regex="", search_folder="", recursive=False, return_multiple=False):

    """ Takes an input filename and returns some output file(s) that match certain
        criteria relative to the input file. Useful for constructing pipelines.

        TODO: Make functional for lists of str.
        TODO: Make output only one type. Propagate changes to rest of package.

        Parameters
        ----------

        input_filename: str
            The original file, which will be referenced to find a linked file(s).
        prefix: str or list of str
            A string prefix or prefixes that the linked file must contain.
        suffix: str or list of str
            A string suffix or suffixes that the linked file must contian.
        includes: str or list of str
            String(s) that must be included in linked file. Linked files are
            those which contain any linked string.
        linux_regex: str
            A linux-style search pattern which must be matched to return
            linked files. Will search on the whole filepath.
        search_folder: str
            Where to look for linked files. Same folder as original file by
            default.
        recursive: bool
            Whether or not to descend down into subfolders when searching.
        return_multiple: bool
            Whether or not to return multiple files if matched. If False,
            will only return the first file and print a warning if multiple
            files are found.

        Returns
        -------
        output: str or list
            Returns linked file or list of files if found.
    """

    if search_folder == "":
        search_folder = os.path.abspath(os.path.dirname(input_filename))

    if recursive:
        file_list = grab_files_recursive(search_folder)
    else:
        file_list = glob.glob(os.path.join(search_folder, '*'))

    output = []

    for file in file_list:

        match = False

        file_basename = os.path.basename(file)

        if prefix != "":
            if file_basename.startswith(prefix):
                match = True

        if suffix != "":
            if file_basename.endswith(suffix):
                match = True
            else:
                match = False

        if includes != "":
            if includes in file_basename:
                match = True
            else:
                match = False

        if linux_regex != "":
            if fnmatch.fnmatch(file, linux_regex):
                match = True
            else:
                match = False

        if match:
            output += [file]

    if len(output) == 1:
        return output[0]
    elif output == [] or return_multiple:
        return output
    else:
        print('Warning: multiple files found. return_multiple is set to False, so only the first file will be returned.')
        return output[0]

def replace_suffix(input_filepath, input_suffix, output_suffix, suffix_delimiter=None):

    """ Replaces an input_suffix in a filename with an output_suffix. Can be used
        to generate or remove suffixes by leaving one or the other option blank.

        TODO: Make suffixes accept regexes. Can likely replace suffix_delimiter after this.
        TODO: Decide whether suffixes should extend across multiple directory levels.

        Parameters
        ----------
        input_filepath: str
            The filename to be transformed.
        input_suffix: str
            The suffix to be replaced
        output_suffix: str
            The suffix to replace with.
        suffix_delimiter: str
            Optional, overrides input_suffix. Replaces whatever 
            comes after suffix_delimiter with output_suffix.

        Returns
        -------
        output_filepath: str
            The transformed filename
    """

    split_filename = nifti_splitext(input_filepath)

    if suffix_delimiter is not None:
        input_suffix = str.split(split_filename[0], suffix_delimiter)[-1]

    if input_suffix not in os.path.basename(input_filepath):
        print('ERROR!', input_suffix, 'not in input_filepath.')
        return []

    else:
        if input_suffix == '':
            prefix = split_filename[0]
        else:
            prefix = input_suffix.join(str.split(split_filename[0], input_suffix)[0:-1])
        prefix = prefix + output_suffix
        output_filepath = prefix + split_filename[1]
        return output_filepath

def sanitize_filename(filename, allowed=[]):

    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in allowed]).rstrip()

def human_sort(l):

    """ Stolen from Stack Exchange. Sorts alphabetically, but also numerically. How?
        who knows.. Maybe it doesn't even work. TODO: test it.
    """

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
    l.sort( key=alphanum )
    return l

def run_test():
    return

if __name__ == '__main__':
    run_test()
