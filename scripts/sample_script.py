from os.path import dirname, realpath
import sys
dir_path = dirname(realpath(__file__))
sys.path += [dir_path, dirname(dir_path)]

import numpy as np
from matplotlib import pyplot as plt

from fluomark.kinetics import monod, monod_inverse
from fluomark.mpl_utils import set_mpl_global, legend


# function definitions


if __name__ == "__main__":
    PLOT_DIR = "plots"
    set_mpl_global(output_dir=PLOT_DIR)

    # script goes here