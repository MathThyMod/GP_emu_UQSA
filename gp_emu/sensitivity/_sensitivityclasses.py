### for the underlying sensistivity classes

import numpy as np
import matplotlib.pyplot as plt

class Sensitivity:
    def __init__(self, emul, v, m):
        print("This is the Sensitivity class being initialised")

        ## inputs stuff
        self.v = v
        self.m = m
        self.x = emul.training.inputs
        
        ## try to use exact values on the MUCM site
        #if True:
        if False:
            emul.par.delta = [[[ 0.5437, 0.0961 ]]]
            emul.par.sigma[0][0] = np.sqrt(0.9354)
            emul.par.sigma[0][0] = np.sqrt(0.92439104)
            emul.par.beta = np.array([ 33.5981 , 4.8570 , -39.6695 ])
            emul.training.remake()

        #### init B
        self.B = np.linalg.inv(np.diag(self.v))
        print("B matrix:\n", self.B)

        #### init C
        self.C = np.diag( 1.0/(np.array(emul.par.delta[0][0])**2) )
        print("C matrix:\n", self.C)

        #### save these things here for convenience
        self.f = emul.training.outputs
        self.H = emul.training.H
        self.beta = emul.par.beta
        self.sigma = emul.par.sigma[0][0] ## only taking the first sigma
        self.A = emul.training.A ## my A has sigma**2 absorbed into it...

        #### calculation and plotting of main effects across range ####
        #### task is to populate self.effect and self.senseindex
        points = 21
        self.effect = np.zeros([self.m.size , points])
        self.senseindex = np.zeros([self.m.size])
        self.EVTw = np.zeros([self.m.size])

        for P in [0,1]:
            print("Sensitivity measures for input", P)
            self.w = [P]
            j = 0 ## j just counts index for each value of xw we try
            for self.xw in np.linspace(0.0,1.0,points): ## changes value of xw
                #self.xw = [i]
                
                self.wb = []
                for k in range(0,len(emul.par.delta[0][0])):
                    if k not in self.w:
                        self.wb.append(k)
                #print("wb:",self.wb)

                self.main_effect(j)
                j=j+1

            self.sensitivity(P) ## sensitivity index doesn't depend on xw
            ## configure plot of main effect of input P
            plt.plot( np.linspace(0.0,1.0,points), self.effect[P] ,\
                linewidth=2.0, label='x'+str(P) )

        for P in [0,1]:
            self.w = [P]
            self.wb = []
            for k in range(0,len(emul.par.delta[0][0])):
                if k not in self.w:
                    self.wb.append(k)
            ### calculate the total effect variance
            self.totaleffectvariance(P)
            

        ## plot a graph of the main effect again xw for all indices
        if points > 1:
            plt.legend(loc='best')
            plt.show()
    
        #### calculation of the sensitivity indices ####

    def main_effect(self, i):
        self.UPSQRT(self.w , self.xw)

        ## have to compensate for MUCM def of A
        invA = np.linalg.inv(self.A/(self.sigma**2))

        #self.e = invA.dot(self.f - self.H.dot(self.beta))
        self.e = np.linalg.solve(self.A/(self.sigma**2), (self.f - self.H.dot(self.beta)) )
            
        self.Emw = self.Rw.dot(self.beta) + self.Tw.dot(self.e)
        self.ME = (self.Rw-self.R).dot(self.beta) + (self.Tw-self.T).dot(self.e)
        #print("xw:",self.xw,"ME_",self.w,":",self.ME)
        self.effect[self.w, i] = self.ME
        ## main effect is giving the correct results


    def sensitivity(self, P):
        self.UPSQRT(P, self.m[P])
        #self.W = np.linalg.inv( (self.H.T).dot(invA).dot(self.H) )
        self.W = np.linalg.inv( (self.H.T).dot(np.linalg.solve(self.A/(self.sigma**2), self.H)  ) )
        #print("W:", self.W)

        self.EEE = (self.sigma**2) *\
             (\
                 self.Uw - np.trace(\
                     np.linalg.solve(self.A, self.Pw) )\
                 +   np.trace(self.W.dot(\
                     self.Qw - self.Sw.dot(np.linalg.solve(self.A, self.H)) -\
                     self.H.T.dot(np.linalg.solve(self.A, self.Sw.T)) +\
                     self.H.T.dot(np.linalg.solve(self.A, self.Pw))\
                     .dot(np.linalg.solve(self.A, self.H))\
                                        )\
                             )\
             )\
             + (self.e.T).dot(self.Pw).dot(self.e)\
             + 2.0*(self.beta.T).dot(self.Sw).dot(self.e)\
             + (self.beta.T).dot(self.Qw).dot(self.beta)

        self.EE2 = (self.sigma**2) *\
             (\
                 self.U - self.T.dot(np.linalg.solve(self.A, self.T.T)) +\
                 ( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)) ) )\
                 .dot( self.W )\
                 .dot( (self.R - self.T.dot(np.linalg.solve(self.A,self.H)).T ))\
             )\
             + ( self.R.dot(self.beta) + self.T.dot(self.e) )**2

        self.EV = self.EEE - self.EE2
        print("xw:",self.xw,"E(V_",self.w,"):",self.EV)
        self.senseindex[P] = self.EV
        #print("EEE:" , self.EEE)
        #print("EE2:" , self.EE2)

        ## find the problems in P, S, Q and W
        ## T and R must be correct because the other answers are correct...


    def totaleffectvariance(self, P):

        self.EVf = (self.sigma**2) *\
            (\
                 self.U - self.T.dot( np.linalg.solve(self.A, self.T.T) ) +\
                 ((self.R - self.T.dot( np.linalg.solve(self.A,self.H) ))\
                 .dot(self.W)\
                 .dot((self.R - self.T.dot(np.linalg.solve(self.A, self.H))).T )\
                 )\
            )

        #print("EVf:", self.EVf)

        self.EVTw[self.w] = self.EVf - self.senseindex[self.wb]
        print("EVT" , P , ":" , self.EVTw[P])

    ### create UPSQRT for particular w and xw
    def UPSQRT(self, w, xw):

        ############# Tw #############
        self.T  = np.zeros([self.x[:,0].size])
        self.Tw = np.zeros([self.x[:,0].size])
        T1 = np.sqrt( self.B.dot(np.linalg.inv(self.B + 2.0*self.C)) ) 
        T2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        T3 = (self.x - self.m)**2
 
        Cww = np.diag(np.diag(self.C)[self.w])
        for k in range(0, self.x[:,0].size):
            #self.T[k]  = np.prod( (T1.dot(np.exp(-T2.dot(T3[k]))))[self.wb] )
            self.T[k]  = np.prod( (T1.dot(np.exp(-T2.dot(T3[k])))) )
            val  = np.prod( (T1.dot(np.exp(-T2.dot(T3[k]))))[self.wb] )
            self.Tw[k] = val\
              *np.exp(-0.5*(xw-self.x[k][self.w]).T.dot(2.0*Cww).dot(xw-self.x[k][self.w]))
            #print("Tw*:\n" , np.exp(-0.5*(xw-self.x[k][w]).T.dot(2.0*Cww).dot(xw-self.x[k][w])) )
        #print("Cww:",Cww)

        #print("T:\n" , self.T)
        #print("Tw:\n" , self.Tw)

        ############# Rw #############
        self.R  = np.append([1.0], self.m)
        Rwno1 = np.array(self.m)
        #print(Rwno1)
        Rwno1[self.w] = xw
        self.Rw = np.append([1.0], Rwno1)
        #print("R:", self.R)
        #print("Rw:", self.Rw)


        ############# Qw #############
        self.Q  = np.outer(self.R.T, self.R)
        self.Qw = np.zeros( [1+len(self.w+self.wb) , 1+len(self.w+self.wb)] )
        # fill in 1
        self.Qw[0][0] = 1.0
        # fill first row and column
        for i in self.wb + self.w:
            self.Qw[0][1+i] = self.m[i]
            self.Qw[1+i][0] = self.m[i]
        
        mwb_mwb = np.outer( self.m[self.wb], self.m[self.wb].T )
        #print( "m(wb)m(wb)^T :", mwb_mwb )
        for i in range(0,len(self.wb)):
            for j in range(0,len(self.wb)):
                self.Qw[1+self.wb[i]][1+self.wb[j]] = mwb_mwb[i][j]
        
        mwb_mw = np.outer( self.m[self.wb], self.m[self.w].T )
        #print( "m(wb)m(w)^T :", mwb_mw )
        for i in range(0,len(self.wb)):
            for j in range(0,len(self.w)):
                self.Qw[1+self.wb[i]][1+self.w[j]] = mwb_mw[i][j]

        mw_mwb = np.outer( self.m[self.w], self.m[self.wb].T )
        #print( "m(w)m(wb)^T :", mw_mwb )
        for i in range(0,len(self.w)):
            for j in range(0,len(self.wb)):
                self.Qw[1+self.w[i]][1+self.wb[j]] = mw_mwb[i][j]

        mw_mw = np.outer( self.m[self.w] , self.m[self.w].T )
        Bww = np.diag( np.diag(self.B)[self.w] )
        mw_mw_Bww = mw_mw + np.linalg.inv(Bww)
        #print( "m(w)m(w)^T + invBww :", mw_mw_Bww )
        for i in range(0,len(self.w)):
            for j in range(0,len(self.w)):
                self.Qw[1+self.w[i]][1+self.w[j]] = mw_mw_Bww[i][j]
        #print("Q:\n",self.Q)
        #print("Qw:\n",self.Qw)


        ############# Sw #############
        self.S  = np.outer(self.R.T, self.T)
        self.Sw = np.zeros( [ 1+len(self.w + self.wb) , self.x[:,0].size ] )
        S1 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 2.0*self.C) ) ) 
        S2 = 0.5*(2.0*self.C*self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        S3 = (self.x - self.m)**2

        for k in range( 0 , 1+len(self.w + self.wb) ):
            for l in range( 0 , self.x[:,0].size ):
                if k == 0:
                    E_star = 1.0
                else:
                    kn=k-1
                    if k-1 in self.wb:
                        E_star = self.m[kn]
                    if k-1 in self.w:
                        E_star=(2*self.C[kn][kn]*self.x[l][kn]\
                               +self.B[kn][kn]*self.m[kn])\
                               /( 2*self.C[kn][kn] + self.B[kn][kn] )
                self.Sw[k,l]=E_star*np.prod( S1.dot( np.exp(-S2.dot(S3[l])) ) )
        #print("Sw:", self.Sw)

        ############# Pw #############
        self.P  = np.outer(self.T.T, self.T)
        self.Pw = np.zeros([self.x[:,0].size , self.x[:,0].size])
        P1 = self.B.dot( np.linalg.inv(self.B + 2.0*self.C) )
        P2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        P3 = (self.x - self.m)**2
        P4 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 4.0*self.C) ) )
        P5 = 0.5*np.linalg.inv(self.B + 4.0*self.C)

        for k in range( 0 , self.x[:,0].size ):
            for l in range( 0 , self.x[:,0].size ):
                P_prod = np.exp(-P2.dot( P3[k]+P3[l] ))
                self.Pw[k,l]=\
                    np.prod( (P1.dot(P_prod))[self.wb] )*\
                    np.prod( (P4.dot(\
                        np.exp( -P5.dot(\
                        4.0*(self.C*self.C).dot( (self.x[k]-self.x[l])**2 )\
                        +2.0*(self.C*self.B).dot(P3[k]+P3[l])) ) ))[self.w] )
        #print("P:" , self.P)
        #print("Pw:" , self.Pw)


        ############# Uw #############
        self.U  = np.prod(np.diag( \
                np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) ) ))
        self.Uw = np.prod(np.diag( \
                np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) ))[self.wb])
        #print("U:", self.U, "Uw:", self.Uw)


