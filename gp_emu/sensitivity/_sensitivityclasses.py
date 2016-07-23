### for the underlying sensistivity classes

import numpy as np

class Sensitivity:
    def __init__(self, emul, v, m):
        print("This is the Sensitivity class being initialised")

        ## inputs stuff
        self.v = v
        self.m = m
        self.x = emul.training.inputs

        #### init B
        self.B = np.linalg.inv(np.diag(self.v))
        print("B matrix:\n", self.B)

        #### init C
        self.C = np.diag( 1.0/(np.array(emul.par.delta[0][0])**2) )
        print("C matrix:\n", self.C)

        self.UPSQRT()

    
    ### then need a class function that create UPSQRT for particular w or {i,j}
    def UPSQRT(self):
        self.U = np.linalg.det(\
                     np.sqrt(\
                         self.B.dot(\
                             np.linalg.inv(self.B + 4.0*self.C)\
                         )\
                     )\
                 )
        print("U:", self.U)

        T1 =\
                     np.sqrt(\
                         self.B.dot(\
                             np.linalg.inv(self.B + 2.0*self.C)\
                         )\
                     ) 
        print(T1)        

        T2 =\
                     2.0*self.C.dot(self.B).dot(\
                         np.linalg.inv(self.B + 2.0*self.C)\
                     )
        print(T2)
 
        T3 = 0.5*(self.x - self.m)**2
     
        print( T2.dot(T3[0]) )
        print( (np.exp(T2.dot(T3[0]))) )
        print( T1.dot(np.exp(T2.dot(T3[0]))) )
 
        self.T = np.zeros([self.x[:,0].size])
        print(self.T)
        for k in range(0, self.x[:,0].size):
            self.T[k]=np.linalg.det( np.diag(T1.dot( np.exp( T2.dot(T3[k]) ) ) ))
        print(self.T)
