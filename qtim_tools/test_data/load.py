""" A convenient function for loading test data store in the qtim_tools
    package. In the future, we may not want to store this data, some of
    which can be huge, in the package, and instead out to download it
    from a separate online hosting point.
"""

import os

def load_test_file(data, return_type="filepath", retrieval_type="local"):

    """ Loads test files. In the future, should download them from an online
        repository. In this case, returning a filepath might be interesting...
    """

    if data == "dce_tofts_v6":
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),'test_data_dce','tofts_v6.nii.gz'))
    elif data == "dce_tofts_v6_label":
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),'test_data_dce','tofts_v6-label.nii.gz'))
    elif data == "dce_tofts_v9":
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),'test_data_dce','tofts_v9.nii.gz'))
    elif data == "dce_tofts_v9_label":
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),'test_data_dce','tofts_v9-label.nii.gz'))
    elif data == "dce_tofts_v9_aif":
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__),'test_data_dce','tofts_v9-AIF-label.nii.gz'))
    else:
        print 'There is no test data under this name. Returning an empty string.'
        return []


    if return_type == "filepath":
        return filepath

if __name__ == '__main__':
    pass