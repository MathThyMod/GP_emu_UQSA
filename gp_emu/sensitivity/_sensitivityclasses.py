### for the underlying sensistivity classes

import numpy as np

class Sensitivity:
    def __init__(self, emul):
        print("This is the Sensitivity class being initialised")

        #### setup B
        ## will need to configure post to predict more points (i.e. so the emulator is interpolating the model across a finer mesh) before doing the sensitivity analysis -- I'll want a function separate from plot() that does this (all the functions are setup, but they're only called in plot() at the momenet
        print("B matrix:")
        self.B = np.diag(emul.post.newnewvar)
        print(self.B)

        #### setup C
        print("C matrix:")
        self.C = np.diag( 1.0/(np.array(emul.par.delta[0][0])**2) )
        print(self.C)

    ### then need a class function that create UPSQRT for particular w or {i,j}
