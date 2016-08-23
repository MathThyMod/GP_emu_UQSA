### for the underlying sensistivity classes

import numpy as np
import matplotlib.pyplot as plt

class Sensitivity:
    def __init__(self, emul, m, v):
        #print("This is the Sensitivity class being initialised")

        ## inputs stuff
        self.v = v
        self.m = m
        self.x = emul.training.inputs
        
        #### init B
        self.B = np.linalg.inv(np.diag(self.v))
        #print("B matrix:\n", self.B)

        #### init C
        self.C = np.diag( 1.0/(np.array(emul.par.delta[0][0])**2) )
        #print("C matrix:\n", self.C)

        #### save these things here for convenience
        self.f = emul.training.outputs
        self.H = emul.training.H
        self.beta = emul.par.beta
        self.sigma = emul.par.sigma[0][0] ## only taking the first sigma
        self.A = emul.training.A ## my A has sigma**2 absorbed into it...
        self.nugget = emul.training.K.nugget[0]
        #print("nugget:", self.nugget)

        #### calculate the unchanging matrices (not dep. on w)
        self.UPSQRT_const()
       
        #### calculate some other unchanging quantities
        self.e = np.linalg.solve(\
            self.A/(self.sigma**2), (self.f - self.H.dot(self.beta)) )
        self.W = np.linalg.inv( (self.H.T)\
            .dot(np.linalg.solve(self.A/(self.sigma**2), self.H)  ) )
        self.G = np.linalg.solve(self.A/(self.sigma**2), self.H)

        ### points for plotting
        #self.points = 5

        ### for saving to file -- set to true when functions have run
        self.done_uncertainty = False
        self.done_main_effect = False
        self.done_sensitivity = False


    def uncertainty(self):
        self.done_uncertainty = True
        ### for the uncertainty analysis ###
        #### some of this code will work in w is not the complete set of inputs
        #### but this is not guaranteed, so for now w is the entire set
        self.w = [i for i in range(0,len(self.m))]
        #print("w:" , self.w)

        ############# R integrals #############
        self.Rh = np.append([1.0], np.array(self.m[self.w]))
        #print("Rh:" , self.Rh)

        self.Rhh = np.zeros([ 1+len(self.w) , 1+len(self.w) ])
        self.Rhh[0][0] = 1.0
        # fill first row and column
        for i in self.w:
            self.Rhh[0][1+i] = self.m[i]
            self.Rhh[1+i][0] = self.m[i]

        mw_mw = np.outer( self.m[self.w] , self.m[self.w].T )
        Bww = np.diag( np.diag(self.B)[self.w] )
        mw_mw_Bww = mw_mw + np.linalg.inv(Bww)
        #print( "m(w)m(w)^T + invBww :", mw_mw_Bww )
        for i in range(0,len(self.w)):
            for j in range(0,len(self.w)):
                self.Rhh[1+self.w[i]][1+self.w[j]] = mw_mw_Bww[i][j]
        #print("Rhh:\n",self.Rhh)

        ## !!!! code currently only works when self.w is complete set !!!!
        self.Rt = np.zeros([self.x[:,0].size])
        self.Rht = np.zeros([1+len(self.w) , self.x[:,0].size])
        for k in range(0, self.x[:,0].size):
            mpk = np.linalg.solve(\
            2.0*self.C+self.B , 2.0*self.C.dot(self.x[k]) + self.B.dot(self.m) )
            #print("m'k:\n" , mpk)
            Qk = 2.0*(mpk-self.x[k]).T.dot(self.C).dot(mpk-self.x[k])\
                  + (mpk-self.m).T.dot(self.B).dot(mpk-self.m)
            self.Rt[k] = (1.0-self.nugget)*np.sqrt(\
                np.linalg.det(self.B)/np.linalg.det(2.0*self.C+self.B))*\
                np.exp(-0.5*Qk)
            Ehx = np.append([1.0], mpk) ## same as Richard's code
            self.Rht[:,k] = self.Rt[k] * Ehx
        #print("Rt:\n" , self.Rt)
        #print("Rht:\n" , self.Rht)

        
        self.Rtt = np.zeros([self.x[:,0].size , self.x[:,0].size])
        for k in range(0, self.x[:,0].size):
            for l in range(0, self.x[:,0].size):
                mpkl = np.linalg.solve(\
                    4.0*self.C+self.B ,\
                    2.0*self.C.dot(self.x[k]) + 2.0*self.C.dot(self.x[l])\
                    + self.B.dot(self.m) )
                Qkl = 2.0*(mpkl-self.x[k]).T.dot(self.C).dot(mpkl-self.x[k])\
                    + 2.0*(mpkl-self.x[l]).T.dot(self.C).dot(mpkl-self.x[l])\
                    + (mpkl-self.m).T.dot(self.B).dot(mpkl-self.m)
                self.Rtt[k,l] = ((1.0-self.nugget)**2)*np.sqrt(\
                    np.linalg.det(self.B)/np.linalg.det(4.0*self.C+self.B))*\
                    np.exp(-0.5*Qkl)
        #print("Rtt:\n" , self.Rtt)


        ############# U integrals #############
        num=len(self.m)
        Bbold = np.zeros([2*num , 2*num])
        Bbold[0:num, 0:num] = 2.0*self.C+self.B
        Bbold[num:2*num, num:2*num] = 2.0*self.C+self.B
        Bbold[0:num, num:2*num] = -2.0*self.C
        Bbold[num:2*num, 0:num] = -2.0*self.C
        #print("Bbold:\n", Bbold)

        self.U2 = (1.0-self.nugget)*\
            np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bbold))
        ## don't know what the actual expected values are ... using R for now
        self.Uh = self.U2 * self.Rh
        self.Uhh = self.U2 * self.Rhh 
    
        Bboldk = np.zeros([2*num , 2*num])
        Bboldk[0:num, 0:num] = 2.0*self.C+self.B
        Bboldk[num:2*num, num:2*num] = 4.0*self.C+self.B
        Bboldk[0:num, num:2*num] = -2.0*self.C
        Bboldk[num:2*num, 0:num] = -2.0*self.C
        self.Ut = np.zeros([self.x[:,0].size])
        self.Uht = np.zeros([1+2*len(self.w) , self.x[:,0].size])
        Ufact = ((1.0-self.nugget)**2)*\
            np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bboldk))
        for k in range(0, self.x[:,0].size):
            mpkvec = np.append( (self.B.dot(self.m)).T ,\
                (2.0*self.C.dot(self.x[k]) + self.B.dot(self.m)).T )
            #print("mpkvec:\n" , mpkvec)
            mpk = np.linalg.solve(Bboldk, mpkvec)
            mpk1 = mpk[0:len(self.m)]
            mpk2 = mpk[len(self.m):2*len(self.m)]
            #print("m'k:\n" , mpk)
            Qku = 2.0*(mpk2-self.x[k]).T.dot(self.C).dot(mpk2-self.x[k])\
                + 2.0*(mpk1-mpk2).T.dot(self.C).dot(mpk1-mpk2)\
                + (mpk1-self.m).T.dot(self.B).dot(mpk1-self.m)\
                + (mpk2-self.m).T.dot(self.B).dot(mpk2-self.m)
            self.Ut[k] = Ufact * np.exp(-0.5*Qku)
            Ehx = np.append([1.0], mpk1) ## again, not sure of value...
            Ehx = np.append(Ehx, mpk2)
            #print("***EHX***:" , Ehx)
            self.Uht[:,k] = self.Ut[k] * Ehx
        #print("Ut:\n" , self.Ut)
        #print("Uht:\n" , self.Uht)

        Bboldkl = np.zeros([2*num , 2*num])
        Bboldkl[0:num, 0:num] = 4.0*self.C+self.B
        Bboldkl[num:2*num, num:2*num] = 4.0*self.C+self.B
        Bboldkl[0:num, num:2*num] = -2.0*self.C
        Bboldkl[num:2*num, 0:num] = -2.0*self.C
        self.Utt = np.zeros([self.x[:,0].size , self.x[:,0].size])
        Ufact2 = ((1.0-self.nugget)**3)*np.linalg.det(self.B)/np.sqrt(np.linalg.det(Bboldkl))
        for k in range(0, self.x[:,0].size):
            mpk = np.linalg.solve(\
                2.0*self.C+self.B , 2.0*self.C.dot(self.x[k])+self.B.dot(self.m) )
            Qk = 2.0*(mpk-self.x[k]).T.dot(self.C).dot(mpk-self.x[k])\
                  + (mpk-self.m).T.dot(self.B).dot(mpk-self.m)
            for l in range(0, self.x[:,0].size):
                mpl = np.linalg.solve(\
                    2.*self.C+self.B,2.*self.C.dot(self.x[l])+self.B.dot(self.m))
                Ql = 2.0*(mpl-self.x[l]).T.dot(self.C).dot(mpl-self.x[l])\
                      + (mpl-self.m).T.dot(self.B).dot(mpl-self.m)

                self.Utt[k,l] = Ufact2 * np.exp(-0.5*(Qk+Ql))
        #print("Utt:\n" , self.Utt)

        self.Utild = 1

        ############# S integrals #############

        Smat = np.zeros([3*num , 3*num])
        Smat[0:num, 0:num] = 4.0*self.C+self.B
        Smat[num:2*num, num:2*num] = 2.0*self.C+self.B
        Smat[2*num:3*num, 2*num:3*num] = 2.0*self.C+self.B
        Smat[0:num, num:2*num] = -2.0*self.C
        Smat[0:num,2*num:3*num] = -2.0*self.C
        Smat[num:2*num, 0:num] = -2.0*self.C
        Smat[2*num:3*num, 0:num] = -2.0*self.C
        #print("Smat:\n" , Smat)
        Smat2 = np.zeros([2*num , 2*num])
        Smat2[0:num, 0:num] = 4.0*self.C+self.B
        Smat2[num:2*num, num:2*num] = 4.0*self.C+self.B
        Smat2[0:num, num:2*num] = -4.0*self.C
        Smat2[num:2*num, 0:num] = -4.0*self.C
        #print("Smat2:\n" , Smat2)

        self.S = ((1.0-self.nugget)**2)*((np.sqrt(np.linalg.det(self.B)))**3)/\
            np.sqrt(np.linalg.det(Smat))
        self.Stild = (1.0-self.nugget)*np.linalg.det(self.B)/\
            np.sqrt(np.linalg.det(Smat2))
        #print("S:\n" , self.S)
        #print("Stild:\n" , self.Stild)

        
        ############# the uncertainty measures #############
        #print("U2:" , self.U2, "U:", self.U) ## values are same
        s2 = (self.sigma**2)
        self.uE = self.Rh.T.dot(self.beta) + self.Rt.T.dot(self.e)
        self.uV = s2*(self.U2-self.Rt.T.dot(np.linalg.solve(self.A/s2,self.Rt))\
            +(self.Rh - self.G.T.dot(self.Rt)).T.dot(self.W)\
            .dot(self.Rh - self.G.T.dot(self.Rt)) )
        self.I1 = s2*(\
            self.Utild - np.trace(np.linalg.solve(self.A/s2,self.Rtt))\
            + np.trace(self.W.dot(self.Rhh-2.0*self.Rht.dot(self.G)+\
            self.G.T.dot(self.Rtt).dot(self.G) ))
                     )
        self.I2 = self.beta.T.dot(self.Rhh).dot(self.beta)\
            + 2.0*self.beta.T.dot(self.Rht).dot(self.e)\
            + self.e.T.dot(self.Rtt).dot(self.e)

        self.uEV = (self.I1-self.uV) + (self.I2 -self.uE**2)
        
        print("\n*** Uncertainty measures ***")
        print("E*[ E[f(X)] ]:",self.uE)
        print("var*[ E[f(X)] ]:",self.uV)
        print("E*[ var[f(X)] ]:",self.uEV)


    def main_effect(self, plot = False, points = 5):
        self.done_main_effect = True
        # for storing the effect
        #points = self.points
        self.effect = np.zeros([self.m.size , points])

        #### initialise the w matrices
        self.Tw=np.zeros([self.x[:,0].size])
        self.Rw=np.zeros([1+1])
        self.Qw=np.zeros([1+len(self.m) , 1+len(self.m)])
        self.Sw=np.zeros([1+len(self.m) , self.x[:,0].size ])
        self.Pw=np.zeros([self.x[:,0].size , self.x[:,0].size])
        self.Uw=0.0
        self.Estar = np.zeros([1+len(self.m),self.x[:,0].size])
        self.P_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.P_b4_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.Uw_b4_prod = np.zeros([len(self.m)])
        
        ### dependance on self.w and self.wb comes later
        self.P_prod_calc()
        self.Uw_calc()

        print("\n*** Main effect measures ***")
        for P in range(0,len(self.m)):
            print("Main effect measures for input", P)
            self.w = [P]
            self.wb = []
            for k in range(0,len(self.m)):
                if k not in self.w:
                    self.wb.append(k)

            self.Qw_calc()
            self.Estar_calc()

            j = 0 ## j just counts index for each value of xw we try
            for self.xw in np.linspace(0.0,1.0,points): ## changes value of xw

                self.UPSQRT(self.w , self.xw)

                self.Emw = self.Rw.dot(self.beta) + self.Tw.dot(self.e)
                self.ME = (self.Rw-self.R).dot(self.beta)\
                    + (self.Tw-self.T).dot(self.e)
                #print("xw:",self.xw,"ME_",self.w,":",self.ME)
                self.effect[P, j] = self.ME
                j=j+1 ## calculate for next xw value
            
            if plot:
                plt.plot( np.linspace(0.0,1.0,points), self.effect[P] ,\
                    linewidth=2.0, label='x'+str(P) )
        if plot:
            plt.legend(loc='best')
            print("Plotting main effects...")
            plt.show()


    #### this needs sorting out, which means UPSQRT needs generalising
    def interaction_effect(self, i, j, points = 5):
        # for storing the interaction - need 2D matrix
        ## currently just storing for a single particular {i,j}
        #points = self.points ## must be the same number as done for main effect...
        self.interaction = np.zeros([points , points])

        ### need to calculate I_{i,j}
        self.w = [i, j]
        self.wb = []
        for k in range(0,len(self.m)):
            if k not in self.w:
                self.wb.append(k)

        #### initialise the w matrices, again...
        self.Tw=np.zeros([self.x[:,0].size])
        self.Rw=np.zeros([1+1])
        self.Qw=np.zeros([1+len(self.m) , 1+len(self.m)])
        self.Sw=np.zeros([1+len(self.m) , self.x[:,0].size ])
        self.Pw=np.zeros([self.x[:,0].size , self.x[:,0].size])
        self.Uw=0.0

        icount = 0 # counts index for each value of xwi we try
        for xwi in np.linspace(0.0,1.0,points): ## value of xw[i]
            jcount = 0 ## j counts index for each value of xwj we try
            for xwj in np.linspace(0.0,1.0,points): ## value of xw[j]
                self.xw=np.array( [ xwi , xwj ] )
                self.UPSQRT(self.w , self.xw)

                #self.Emw = self.Rw.dot(self.beta) + self.Tw.dot(self.e)
                self.IE = (self.Rw - 3*self.R).dot(self.beta)\
                    + (self.Tw - 3*self.T).dot(self.e)\
                    - self.effect[i, icount]\
                    - self.effect[j, jcount]

                print("xw:",self.xw,"IE_",self.w,":",self.IE)
                self.interaction[icount, jcount] = self.IE
                jcount=jcount+1 ## calculate for next xw value
            icount=icount+1 ## calculate for next xw value

        ## contour plot of interaction effects
        #xplot = np.linspace(0.0,1.0,points)
        #yplot = np.linspace(0.0,1.0,points)
        #ZF = np.zeros((points,points))
        #for i in range(0,points):
        #    for j in range(0,points):
        #        ZF[i,j]=self.interaction[i, j]

        print(self.interaction)

        fig = plt.figure()        
        im = plt.imshow(self.interaction, origin='lower',\
             cmap=plt.get_cmap('rainbow'), extent=(0.0,1.0,0.0,1.0))
        plt.colorbar()
        plt.show()

    def sensitivity(self):
        self.done_sensitivity = True
        print("\n*** Calculate sensitivity indices ***")
        self.senseindex = np.zeros([self.m.size])

        #### initialise the w matrices
        self.Tw=np.zeros([self.x[:,0].size])
        self.Qw=np.zeros([1+len(self.m) , 1+len(self.m)])
        self.Sw=np.zeros([1+len(self.m) , self.x[:,0].size ])
        self.Pw=np.zeros([self.x[:,0].size , self.x[:,0].size])
        ## these get redefined anyway
        self.Rw=np.zeros([1+1]) ## for when w is a single index
        self.Uw=0.0
        self.Estar = np.zeros([1+len(self.m),self.x[:,0].size])
        self.P_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.P_b4_prod = np.zeros([self.x[:,0].size,self.x[:,0].size,len(self.m)])
        self.Uw_b4_prod = np.zeros([len(self.m)])

        ### dependance on self.w and self.wb comes later
        self.P_prod_calc()
        self.Uw_calc()

        for P in range(0,len(self.m)):
            #print("Sensitivity measures for input", P)
            self.w  = [P]
            self.wb = []
            for k in range(0,len(self.m)):
                if k not in self.w:
                    self.wb.append(k)
            self.xw = self.m[P]

            self.Qw_calc()
            self.Estar_calc()

            self.UPSQRT(self.w , self.xw)

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

            self.EVint = self.EEE - self.EE2
            print("E(V" + str(self.w) +"):", self.EVint)
            self.senseindex[P] = self.EVint


    ##### isn't clear that this is correct results, since no MUCM examples...
    def totaleffectvariance(self):
        print("\n*** Calculate total effect variance ***")
        self.senseindexwb = np.zeros([self.m.size])
        self.EVTw = np.zeros([self.m.size])

        #### this is another constant
        #### it's value is tiny since I divided by sigma... maybe sigma problem...
        self.EVf = (self.sigma**2) *\
            (\
                 self.U - self.T.dot( np.linalg.solve(self.A/(self.sigma**2), self.T.T) ) +\
                 ((self.R - self.T.dot( np.linalg.solve(self.A/(self.sigma**2),self.H) ))\
                 .dot(self.W)\
                 .dot((self.R - self.T.dot(np.linalg.solve(self.A/(self.sigma**2), self.H))).T )\
                 )\
            )
        
        print("EVf:", self.EVf)


        for P in range(0,len(self.m)):
            self.w = [P]
            self.wb = []
            for k in range(0,len(self.m)):
                if k not in self.w:
                    self.wb.append(k)

            ## swap around so we calc E*[V_wb]
            temp = self.w
            self.w = self.wb
            self.wb = temp
            ## then define xw as the means (value doesn't matter for sensitivity)
            self.xw = self.m[self.w]

            #### calculate E*[V_wb]
            print(self.w , self.xw)
            self.UPSQRT(self.w , self.xw)

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

            
            self.EVaaa = self.EEE - self.EE2
            print("xw:",self.xw,"E(V_",self.w,"):",self.EVaaa)
            self.senseindexwb[P] = self.EVaaa

            #########################

            print("senseindexwb" , self.w , ":" , self.senseindexwb[P] )

            self.EVTw[P] = self.EVf - self.senseindexwb[P]
            print("EVT" , P , ":" , self.EVTw[P])


    def UPSQRT_const(self):

        ############# T #############
        self.T  = np.zeros([self.x[:,0].size])
        self.T1 = np.sqrt( self.B.dot(np.linalg.inv(self.B + 2.0*self.C)) ) 
        self.T2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.T3 = (self.x - self.m)**2
 
        self.Tk_b4_prod = np.zeros([self.x[:,0].size, len(self.m)])
        #print(self.Tk_b4_prod)

        for k in range(0, self.x[:,0].size):
            self.Tk_b4_prod[k,:]=(self.T1.dot(np.exp(-self.T2.dot(self.T3[k]))))
            self.T[k]= (1.0-self.nugget)*np.prod( self.Tk_b4_prod[k] )
                #np.prod( (self.T1.dot(np.exp(-self.T2.dot(self.T3[k])))) )

        ############# RQSPU #############
        self.R = np.append([1.0], self.m)
        self.Q = np.outer(self.R.T, self.R)
        self.S = np.outer(self.R.T, self.T)
        self.P = np.outer(self.T.T, self.T)
        self.U = (1.0-self.nugget)*np.prod(np.diag(\
                np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) ) ))

        ##### other constant matrices used for RwQw etc.
        self.S1 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 2.0*self.C) ) ) 
        self.S2 = 0.5*(2.0*self.C*self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.S3 = (self.x - self.m)**2
        self.Sw_b4_prod = np.zeros([self.x[:,0].size, len(self.m)])
        for l in range( 0 , self.x[:,0].size ):
            self.Sw_b4_prod[l,:] = self.S1.dot( np.exp(-self.S2.dot(self.S3[l])) )

        self.P1 = self.B.dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.P2 = 0.5*2.0*self.C.dot(self.B).dot( np.linalg.inv(self.B + 2.0*self.C) )
        self.P3 = (self.x - self.m)**2
        self.P4 = np.sqrt( self.B.dot( np.linalg.inv(self.B + 4.0*self.C) ) )
        self.P5 = 0.5*np.linalg.inv(self.B + 4.0*self.C)


    def Qw_calc(self):
        ############# Qw #############
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
        #print("Qw:\n",self.Qw)

    def Estar_calc(self):
        for k in range( 0 , 1+len(self.m) ):
            for l in range( 0 , self.x[:,0].size ):
                if k == 0:
                    self.Estar[k,l] = 1.0
                else:
                    kn=k-1
                    if k-1 in self.wb:
                        self.Estar[k,l] = self.m[kn]
                    if k-1 in self.w:
                        self.Estar[k,l]=(2*self.C[kn][kn]*self.x[l][kn]\
                               +self.B[kn][kn]*self.m[kn])\
                               /( 2*self.C[kn][kn] + self.B[kn][kn] )

    def P_prod_calc(self):
        for k in range( 0 , self.x[:,0].size ):
            for l in range( 0 , self.x[:,0].size ):
                self.P_prod[k,l,:] = np.exp(-self.P2.dot( self.P3[k]+self.P3[l] ))
                self.P_b4_prod[k,l,:] =\
                    (self.P4.dot(\
                        np.exp( -self.P5.dot(\
                        4.0*(self.C*self.C).dot( (self.x[k]-self.x[l])**2 )\
                        +2.0*(self.C*self.B).dot(self.P3[k]+self.P3[l])) ) ))

    def Uw_calc(self):
        self.Uw_b4_prod =\
            np.diag(np.sqrt(self.B.dot(np.linalg.inv(self.B+4.0*self.C))))


    ### create UPSQRT for particular w and xw
    def UPSQRT(self, w, xw):

        ############# Tw #############
        Cww = np.diag(np.diag(self.C)[self.w])
        for k in range(0, self.x[:,0].size):
            #val  = np.prod( (self.T1.dot(np.exp(-self.T2.dot(self.T3[k]))))[self.wb] )
            val  = np.prod( self.Tk_b4_prod[k][self.wb] )
            self.Tw[k] = (1.0-self.nugget)*val\
              *np.exp(-0.5*(xw-self.x[k][self.w]).T.dot(2.0*Cww).dot(xw-self.x[k][self.w]))

        ############# Rw #############
        Rwno1 = np.array(self.m)
        Rwno1[self.w] = self.xw
        self.Rw = np.append([1.0], Rwno1)

        ############# Sw #############

        for k in range( 0 , 1+len(self.m) ):
            for l in range( 0 , self.x[:,0].size ):
                self.Sw[k,l]=(1.0-self.nugget)*self.Estar[k,l]*\
                    np.prod( self.Sw_b4_prod[l] )
                    #np.prod( self.S1.dot( np.exp(-self.S2.dot(self.S3[l])) ) )

        #print("Sw:", self.Sw)

        #self.P_prod_calc()

        ############# Pw #############

        for k in range( 0 , self.x[:,0].size ):
            for l in range( 0 , self.x[:,0].size ):
                #P_prod = np.exp(-self.P2.dot( self.P3[k]+self.P3[l] ))
                self.Pw[k,l]=((1.0-self.nugget)**2)*\
                    np.prod( (self.P1.dot(self.P_prod[k,l]))[self.wb] )*\
                    np.prod( self.P_b4_prod[k,l,self.w] )
        #print("P:" , self.P)
        #print("Pw:" , self.Pw)

        #self.Uw_calc()

        ############# Uw #############
        #self.Uw = (1.0-self.nugget)*np.prod(np.diag( \
        #        np.sqrt( self.B.dot(np.linalg.inv(self.B+4.0*self.C)) ))[self.wb])
        self.Uw = (1.0-self.nugget)*np.prod(self.Uw_b4_prod[self.wb])
        #print("U:", self.U, "Uw:", self.Uw)

        
    def to_file(self, filename):
        print("Sensitivity & Uncertainty results to file...")
        f=open(filename, 'w')

        if self.done_uncertainty == True :
            f.write("EE " + str(self.uE) +"\n")
            f.write("VE " + str(self.uV) +"\n")
            f.write("EV "+ str(self.uEV) +"\n")
        
        if self.done_sensitivity == True :
            f.write("EVw " + ' '.join(map(str,self.senseindex)) +"\n")

        if self.done_main_effect == True :
            f.write("xw "+' '.join(map(str,\
                [i for i in np.linspace(0.0,1.0,self.effect[0].size)] ))+"\n")
            for i in range(0, len(self.m)):
                f.write("ME"+str(i)+" "+ ' '.join(map(str,self.effect[i])) + "\n")
            
        f.close()
