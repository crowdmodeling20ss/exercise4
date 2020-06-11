import os

import numpy as np

MAIN_PATH = os.path.abspath(os.getcwd()) + '/'


def read_file(file_path):
    """
    Read data from a given file then return the numpy array

    :param file_path: name of the file with extension. e.g. pca_dataset.txt
    :return: [N, D] numpy array of the data in the file
    """

    file = open(MAIN_PATH + "../data/" + file_path, "r")
    var = []
    for line in file:
        # TODO: float may cause casting issue. Check it!
        var.append(tuple(map(float, line.rstrip().split())))
    file.close()

    return np.array(var)
