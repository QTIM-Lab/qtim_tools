import numpy as np
import os

from shutil import copy

def Copy_SameAIF_Visit1_Tumors(input_directory):

    """ For one vs. all comparisons, it will be easier to code if we have duplicate
        visit 1 entries for the "SameAIF" mode. Because SameAIF uses the AIF from visit
        1 in both entries, visit 1 should be unchanged with this option.
    """

    visit_1_file_database = glob.glob(os.path.join(input_directory, '*VISIT_01_autoAIF*.nii*'))

    for filename in visit_1_file_database:
        new_filename = str.split(filename, 'VISIT_01')[0] + '_sameAIF_' + str.split(filename, 'VISIT_01')[-1]
        copy(filename, new_filename)




if __name__ == '__main__':
    pass