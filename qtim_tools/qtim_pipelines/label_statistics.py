""" This function is meant to generate ROI-based statistics for QTIM
    volumes.
"""

import nipype.interfaces.io as nio
import os

def qtim_study_statistics(study_name, label_file, output_csv, base_directory='/qtim2/users/data/'):

    print 'Example Command Line Succeeded'

    datasource1 = nio.DataGrabber()
    datasource1.inputs.base_directory = base_directory
    datasource1.inputs.template = os.path.join(study_name, 'ANALYSIS', 'COREGISTRATION', study_name + '*', 'VISIT_*', '*.ni*')
    datasource1.inputs.sort_filelist = True

    results = datasource1.run()
    print results

    return

def run_test():
    return

if __name__ == '__main__':
    run_test()