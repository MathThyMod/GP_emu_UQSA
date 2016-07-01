### for the user accessable sensitivity functions
from ._sensitivityclasses import *

def setup(emul):
    print("\nsetup function for initialising Sensitivity class")
    s = Sensitivity(emul)
