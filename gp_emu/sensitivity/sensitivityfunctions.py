### for the user accessable sensitivity functions
import numpy as _np
from ._sensitivityclasses import *

def setup(emul, case, m, v):
    print("\nsetup function for initialising Sensitivity class")

    m = _np.array(m)
    v = _np.array(v)

    s = Sensitivity(emul, m, v)

    return s
