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

        #### init B
        self.B = np.linalg.inv(np.diag(self.v))
        print("B matrix:\n", self.B)

        #### init C
        self.C = np.diag( 1.0/(np.array(emul.par.delta[0][0])**2) )
        #self.C = np.diag( [ 3.3828 , 108.2812 ] ) ## MUCM values
        print("C matrix:\n", self.C)

        points = 21
        self.effect = np.zeros([points])
        self.xplot = np.linspace(0.0,1.0,points)
        j = 0
        for i in np.linspace(0.0,1.0,points):
            self.w = [1]
            self.xw = [i]
            
            self.wb = []
            for i in range(0,len(emul.par.delta[0][0])):
                if i not in self.w:
                    self.wb.append(i)
            #print("wb:",self.wb)

            self.H = emul.training.H
            self.A = emul.training.A ## my A may be different than MUCM - mine has already absorbed the sigma**2 into it...
            self.f = emul.training.outputs
            self.beta = emul.par.beta
            self.sigma = emul.par.sigma[0][0] ## only taking the first sigma


            ## we want to call these multiple times for different xw, get graph
            self.UPSQRT(self.w , self.xw)
            self.analyse(j)
            j=j+1

        plt.plot(self.xplot, self.effect , linewidth=2.0)
        plt.show()
    

    ### create UPSQRT for particular w and xw
    def UPSQRT(self, w, xw):

        ##########
        ### Tw ###
        ##########

        T1 = np.sqrt( self.B.dot(np.linalg.inv(self.B + 2.0*self.C)) ) 
        #print(T1)        

        T2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        #print(T2)
        ## does this properly capture the k summing?
        T3 = (self.x - self.m)**2
     
        # just the first (i=0) bits are #printed
        ##print( T2.dot(T3[0]) )
        ##print( (np.exp(T2.dot(T3[0]))) )
        ##print( T1.dot(np.exp(T2.dot(T3[0]))) )
        #T_prod = T1.dot(np.exp(T2.dot(T3)))
        ##print("T_prod:",T_prod)
 
        self.T = np.zeros([self.x[:,0].size])
        #print("T:", self.T)
        for k in range(0, self.x[:,0].size):
            #self.T[k]=np.linalg.det( np.diag(T1.dot( np.exp( T2.dot(T3[k]) ) ) ))
            self.T[k]=np.prod( (T1.dot(np.exp(-T2.dot(T3[k]))))[self.wb] )
        #print(self.T)
        
        Cww = np.diag(np.diag(self.C)[self.w])
        self.Tw = np.zeros([self.x[:,0].size])
        for k in range(0, self.x[:,0].size):
            self.Tw[k] = self.T[k]\
            * np.exp( -0.5*\
                ((xw - self.x[k][w]).T.dot(\
                2.0*Cww)).dot(\
                (xw - self.x[k][w])) )

        #print("Tw:", self.Tw)
        

        ##########
        ### Rw ###
        ##########

        self.R = np.append([1.0], self.m)
        #print("R:",self.R)
        Rwno1 = np.array(self.m)
        #print("Rwno1:",Rwno1)
        Rwno1[w] = xw
        #print("Rwno1:",Rwno1)
        self.Rw = np.append([1.0], Rwno1)
        #print("Rw:",self.Rw)

        ##########
        ### Qw ###
        ##########

        self.Q = np.outer(self.R.T, self.R)
        #print("Q:",self.Q)

        self.Qw = np.zeros([1+len(self.w+self.wb),1+len(self.w+self.wb)])
        #print("Qw:",self.Qw)
        # fill in 1
        self.Qw[0,0] = 1.0
        # fill first row
        for i in self.wb:
            self.Qw[0][1+i] = self.m[i]
        for i in self.w:
            self.Qw[0][1+i] = self.m[i]
        # fill first col
        for i in self.wb:
            self.Qw[1+i][0] = self.m[i]
        for i in self.w:
            self.Qw[1+i][0] = self.m[i]
        # m(wb)m(wb)^T
        mwb_mwb = np.outer( self.m[self.wb], self.m[self.wb].T )
        #print( "m(wb)m(wb)^T :", mwb_mwb )
        for i in range(0,len(self.wb)):
            for j in range(0,len(self.wb)):
                self.Qw[1+self.wb[i]][1+self.wb[i]] = mwb_mwb[i][j]
        # m(wb)m(w)^T
        mwb_mw = np.outer( self.m[self.wb], self.m[self.w].T )
        #print( "m(wb)m(w)^T :", mwb_mw )
        for i in range(0,len(self.wb)):
            for j in range(0,len(self.w)):
                self.Qw[1+self.wb[i]][1+self.w[i]] = mwb_mw[i][j]
        # m(w)m(wb)^T
        mw_mwb = np.outer( self.m[self.w], self.m[self.wb].T )
        #print( "m(w)m(wb)^T :", mw_mwb )
        for i in range(0,len(self.w)):
            for j in range(0,len(self.wb)):
                self.Qw[1+self.w[i]][1+self.wb[i]] = mw_mwb[i][j]
        # m(w)m(w)^T + Bww^-1
        mw_mw = np.outer( self.m[self.w], self.m[self.w].T )
    
        Bww = np.diag(np.diag(self.B)[self.w])
        #print("Bww:",Bww)
        mw_mw_Bww = mw_mw + np.linalg.inv(Bww)
        #print( "m(w)m(w)^T + invBww :", mw_mw_Bww )
        for i in range(0,len(self.w)):
            for j in range(0,len(self.w)):
                self.Qw[1+self.w[i]][1+self.w[i]] = mw_mw_Bww[i][j]
        #print("Qw:",self.Qw)


        ##########
        ### Sw ###
        ##########

        self.S = np.outer(self.R.T, self.T)
        #print("S:",self.S)

        S1 =\
                     np.sqrt(\
                         self.B.dot(\
                             np.linalg.inv(self.B + 2.0*self.C)\
                         )\
                     ) 
        ##print(S1)        

        S2 =\
                     2.0*self.C.dot(self.B).dot(\
                         np.linalg.inv(self.B + 2.0*self.C)\
                     )
        ##print(S2)
 
        S3 = 0.5*(self.x - self.m)**2


        ##print("SIZE:",self.x[:,0].size)
        self.Sw = np.zeros( [len(self.w + self.wb) + 1 , self.x[:,0].size] )
        for k in range( 0 , len(self.w + self.wb) + 1 ):
            for l in range( 0 , self.x[:,0].size ):
                ##print("k,l:",k,l)
                if k == 0:
                    E_star = 1.0
                if k in self.wb:
                    E_star = self.m[k]
                if k == self.w:
                    E_star=(2.*self.C[k][k]*self.x[k][l]+self.B[k][k]*self.m[k])/\
                        ( 2.*self.C[k][k] + self.B[k][k] )
                self.Sw[k,l]=E_star*\
                    np.prod( S1.dot( np.exp( S2.dot(S3[l]) ) ) )
                    #np.linalg.det( np.diag(S1.dot( np.exp( S2.dot(S3[l]) ) ) ))

#
#        # just the first (i=0) bits are #printed
#        #print( S2.dot(S3[0]) )
#        #print( (np.exp(S2.dot(S3[0]))) )
#        #print( S1.dot(np.exp(S2.dot(S3[0]))) )
# 
        #print("Sw:", self.Sw)

        ##########
        ### Pw ###
        ##########

        self.P = np.outer(self.T.T, self.T)
        #print("P:",self.P)
        
        self.Pw = np.zeros([self.x[:,0].size , self.x[:,0].size])
        
        P1 = self.B.dot( np.linalg.inv(self.B + 2.0*self.C) )
        ##print(P1)        

        P2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        ##print(P2)
 
        P3 = ( self.x - self.m )**2

        P4 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 4.0*self.C) ) )

        P5 = 0.5*np.linalg.inv(self.B + 4.0*self.C)

        for k in range( 0 , self.x[:,0].size ):
            for l in range( 0 , self.x[:,0].size ):
                P_prod = np.exp(-P2.dot(P3[k])).dot( np.exp(-P2.dot(P3[l])) )
                self.Pw[k,l]=\
                    np.prod( (P1.dot(P_prod))[self.wb] ) *\
                    np.prod( (P4.dot(\
                        np.exp(-P5.dot(\
                        4.*self.C.dot(self.C).dot((self.x[k,i]-self.x[l,i])**2)))\
                        *\
                        np.exp(-P5.dot(\
                        2.0*self.C.dot(self.B).dot(P3[k])))\
                        *\
                        np.exp(-P5.dot(\
                        2.0*self.C.dot(self.B).dot(P3[l])))\
                        ))[self.w] )

        #print("Pw:" , self.Pw)


        ##########
        ### Uw ###
        ##########

#        self.U = np.linalg.det(\
#                     np.sqrt(\
#                         self.B.dot(\
#                             np.linalg.inv(self.B + 4.0*self.C)\
#                         )\
#                     )\
#                 )

        self.U = np.prod(\
                    np.diag(\
                     np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) )\
                    ))
        #print("U:", self.U)

        self.Uw = np.prod(\
                    np.diag(\
                     np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) )\
                    )[self.wb])

        #print("Uw:",self.Uw)


    def analyse(self, i):
        print("This is the analyse function")

        self.e = (np.linalg.inv(self.A/(self.sigma**2))\
            .dot(self.f - self.H.dot(self.beta)))
            

        self.Emw = self.Rw.dot(self.beta) + self.Tw.dot(self.e)
        self.ME = (self.Rw-self.R).dot(self.beta) + (self.Tw-self.T).dot(self.e)
        print("xw:",self.xw,"ME:",self.ME)
        self.effect[i] = self.ME
        ## main effect is giving the correct results

        ## have to compensate for MUCM def of A
        invA = np.linalg.inv(self.A)#*self.sigma**2

        self.W=\
            np.linalg.inv(self.H.T.dot(invA).dot(self.H))#\
#            /self.sigma**2

        self.EEE = self.sigma**2 *\
            (\
                self.Uw - np.trace(invA*self.Pw)\
                +   np.trace(self.W.dot(\
                    self.Qw - self.Sw.dot(invA).dot(self.H) -\
                    self.H.T.dot(invA).dot(self.Sw.T) +\
                    self.H.T.dot(invA).dot(self.Pw).dot(invA).dot(self.H)\
                                    )\
                    )\
            )\
            +\
            self.e.T.dot(self.Pw).dot(self.e) +\
            2.0*self.beta.T.dot(self.Sw).dot(self.e) +\
            self.beta.T.dot(self.Qw).dot(self.beta)

        self.EE2 = self.sigma**2 * \
           (\
                self.U - self.T.dot(invA).dot(self.T.T) +\
                (self.R - self.T.dot(invA).dot(self.H)).dot(self.W).dot(\
                    (self.R - self.T.dot(invA).dot(self.H)).T\
                    )\
           )\
           + (self.R.dot(self.beta) + self.T.dot(self.e))**2

        self.EV = self.EEE - self.EE2
        print("xw:",self.xw,"E(V):",self.EV)

        ## find the problems in P, S, Q and W - T and R must be correct because the other answers are correct...
