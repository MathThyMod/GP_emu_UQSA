### for the user accessable sensitivity functions
import numpy as _np
from ._sensitivityclasses import *

def setup(emul, case):
    print("\nsetup function for initialising Sensitivity class")

    if case == "case2":
        v = [0.02, 0.02]
        m = [0.50, 0.50]

    s = Sensitivity(emul, v, m)
