""" This will be a utility for reading human-readable data passed in through
    text-based formats. These data may be transformation matrices, AIFs, bvectors,
    or other relevant information.
"""

import numpy as np
import csv

def save_numpy_2_csv(input_numpy, output_csv, ignore_value=False):

    """ A convenience function to write a numpy array to a csv
        file. Numpy has a method to do this, but I often find it
        counterintuitive. This also allows you to not write certain
        rows depending on a value in the first column.

        Parameters
        ----------

        input_numpy: array
            An array, preferably of object datatype, to write.
        output_csv: str
            A csv file to write to.
        ignore_value: str or bool
            If ignore_value is in the first column, this row will
            be skipped. Default is set to False, which does not
            ignore any value.
    """

    with open(output_csv, 'wb') as writefile:
        csvfile = csv.writer(writefile, delimiter=',')
        for row in input_numpy:
            if (ignore_value or ignore_value == '') and row[0] == ignore_value:
                continue
            csvfile.writerow(row)

def run_test():
    return

if __name__ == '__main__':
    run_test()