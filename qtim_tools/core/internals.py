""" This module contains shared code by all other modules in
    qtim_tools. They are not necessarily meant to be used outside
    of the context of the package.
"""

class qtim_Exception(Exception):
    
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)



def run_test():
    return

if __name__ == '__main__':
    run_test()