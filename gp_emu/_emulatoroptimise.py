from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


### for optimising the hyperparameters
class Optimize:
    def __init__(self, data, basis, par, beliefs, config):
        self.data = data
        self.basis = basis
        self.par = par
        self.beliefs = beliefs
        self.config = config
        self.standard_constraint()
        
        print("\n*** Optimization options ***")
 
        # if bounds are empty then construct them automatically
        if config.bounds == ():
            bounds_t = []
            for d in range(0, len(self.data.K.delta)):
                for i in range(0,self.data.K.delta[d].size):
                    bounds_t.append([0.001,1.0])
            for s in range(0, len(self.data.K.delta)):
                for i in range(0,self.data.K.sigma[s].size):
                    bounds_t.append([0.001,100.0])
            config.bounds = tuple(bounds_t)
            print("No bounds provided, so setting to default:")
            print(config.bounds)
        else:
            print("User provided bounds:")
            print(config.bounds)

        # set up type of bounds
        if config.constraints_type == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            if config.constraints_type == "noise":
                self.noise_constraint()
            else:
                print("Constraints set to standard")
                self.standard_constraint()
        
    ## tries to keep deltas above a small value
    def standard_constraint(self):
        self.cons = []
        #for i in range(0, self.data.K.delta_num):
        for i in range(0, self.data.K.delta_num + self.data.K.sigma_num):
            hess = np.zeros(self.data.K.delta_num + self.data.K.sigma_num)
            hess[i]=1.0
            dict_entry= {\
                        'type': 'ineq',\
                        'fun' : lambda x, f=i: x[f] - 2.0*np.log(0.001),\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
        self.cons = tuple(self.cons)
        

    ## tries to keep within the bounds as specified for global stochastic opt
    def bounds_constraint(self, bounds):
        print("Setting full bounds constraint")
        self.cons = []
        for i in range(0, self.data.K.delta_num + self.data.K.sigma_num):
            hess = np.zeros(self.data.K.delta_num + self.data.K.sigma_num)
            hess[i] = 1.0
            lower, upper = bounds[i]
            dict_entry = {\
              'type': 'ineq',\
              'fun' : lambda x, a=lower, f=i: x[f] - 2.0*np.log(a),\
              'jac' : lambda x, h=hess: h\
            }
            self.cons.append(dict_entry)
            dict_entry = {\
              'type': 'ineq',\
              'fun' : lambda x, b=upper, f=i: 2.0*np.log(b) - x[f],\
              'jac' : lambda x, h=hess: h\
            }
            self.cons.append(dict_entry)
        self.cons = tuple(self.cons)


    ## tries to keep deltas above a small value, and fixes sigma noise
    def noise_constraint(self):
        # assume last sigma is noise (last kernel supplied is noise)
        # keeps sigma - 0.001 < sigma < sigma + 0.001
        if self.data.K.name[-1] == "Noise" :
            n = self.data.K.delta_num + self.data.K.sigma_num
            noise_val = 2.0*np.log(self.data.K.sigma[-1][0])
            self.cons = []
            hess = np.zeros(n)
            hess[n-1] = 1.0
            dict_entry = {\
              'type': 'ineq',\
              'fun' : lambda x, f=n-1, nv=noise_val: x[f]-(nv-2.0*np.log(0.001)),\
              'jac' : lambda x, h=hess: h\
            }
            self.cons.append(dict_entry)
            dict_entry= {\
                        'type': 'ineq',\
                'fun' : lambda x, f=n-1, nv=noise_val: (nv+2.0*np.log(0.001))-x[f],\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
            for i in range(0,self.data.K.delta_num):
                hess = np.zeros(n)
                hess[i]=1.0
                dict_entry= {\
                            'type': 'ineq',\
                            'fun' : lambda x, f=i: x[f] - 2.0*np.log(0.001),\
                            'jac' : lambda x, h=hess: h\
                            }
                self.cons.append(dict_entry)
            self.cons = tuple(self.cons)
            print("Noise constraint set")
        else:
            print("Last kernel is not Noise, so Noise constraint won't work")


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        use_cons = self.config.constraints
        bounds = self.config.bounds
        stochastic = self.config.stochastic

        print("Optimising delta and sigma...")

        ### scale the provided bounds
        bounds_new = []
        for i in bounds:
            temp = 2.0*np.log(np.array(i))
            bounds_new = bounds_new + [list(temp)]
        bounds = tuple(bounds_new)
        
        ## actual function containing the optimizer calls
        self.optimal(numguesses, use_cons, bounds, stochastic, print_message)

        print("best delta: " , self.par.delta)
        print("best sigma: " , self.par.sigma)
        #print("best sigma**2: ", [[j**2 for j in i] for i in self.par.sigma])

        if self.beliefs.fix_mean == 'F':
            self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals = 4))

   
    def optimal(self,\
      numguesses, use_cons, bounds, stochastic, print_message=False):
        first_try = True
        best_min = 10000000.0

        ## params - number of paramaters that need fitting
        params = self.data.K.delta_num + self.data.K.sigma_num

        ## if MUCM case
        MUCM = False
        if len(self.data.K.name) == 1 and self.data.K.name[0] == "gaussian_mucm":
            print("Gaussian kernel only -- sigma is function of delta")
            MUCM = True
            params = params - 1 # no longer need to optimise sigma

        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## tell user which fitting method is being used
        if stochastic:
            print("Using global stochastic method " 
                  "(diff evol method is bounded)...")
        else:
            if use_cons:
                print("Using constrained COBYLA method...")
            else:
                print("Using Nelder-Mead method...")

        ## try each x-guess (start value for optimisation)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])
            if True:
                if stochastic:
                    while True:
                        if MUCM:
                            res=differential_evolution(self.loglikelihood_mucm,\
                              bounds[0:len(bounds)-1], maxiter=200\
                              )#, tol=0.1)
                        else:
                            res=differential_evolution(self.loglikelihood_gp4ml,\
                              bounds, maxiter=200\
                              )#, tol=0.1)
                        if print_message:
                            print(res, "\n")
                        if res.success == True:
                            break
                        else:
                            print(res.message, "Trying again.")
                else:
                    if use_cons:
                        if MUCM:
                            res = minimize(self.loglikelihood_mucm,\
                              x_guess,constraints=self.cons,\
                                method='COBYLA'\
                                )#, tol=0.1)
                        else:
                            res = minimize(self.loglikelihood_gp4ml,\
                              x_guess,constraints=self.cons,\
                                method='COBYLA'\
                                )#, tol=0.1)
                        if print_message:
                            print(res, "\n")
                    else:
                        if MUCM:
                            res = minimize(self.loglikelihood_mucm,
                              x_guess, method = 'Nelder-Mead'\
                              )#,options={'xtol':0.1, 'ftol':0.001})
                        else:
                            res = minimize(self.loglikelihood_gp4ml,
                              x_guess, method = 'Nelder-Mead'\
                              )#,options={'xtol':0.1, 'ftol':0.001})
                        if print_message:
                            print(res, "\n")
                            if res.success != True:
                                print(res.message, "Not succcessful.")
                print("  result: " , np.around(np.exp(res.x/2.0),decimals=4),\
                      " llh: ", -1.0*np.around(res.fun,decimals=4))
                #print("res.fun:" , res.fun)
                if (res.fun < best_min) or first_try:
                    best_min = res.fun
                    best_x = np.exp(res.x/2.0)
                    best_res = res
                    first_try = False
        print("********")
        if MUCM:
            self.sigma_analytic_mucm(best_x)  ## sets par.sigma correctly
            self.x_to_delta_and_sigma(np.append(best_x , self.par.sigma))
        else:
            self.x_to_delta_and_sigma(best_x)
        ## store these values in par, so we remember them
        self.par.delta = [[list(i) for i in d] for d in self.data.K.delta]
        self.par.sigma = [list(s) for s in self.data.K.sigma]

        self.data.make_A()
        self.data.make_H()


    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x):
        x = np.exp(x/2.0) ## undo the transformation...
        self.x_to_delta_and_sigma(x) ## give values to kernels
        self.data.make_A() ## construct covariance matrix

        ## calculate llh via solver routines - slower, less stable
        if False:
        #start = time.time()
        #for count in range(0,1000):
            (signdetA, logdetA) = np.linalg.slogdet(self.data.A)
            val=linalg.det( ( np.transpose(self.data.H) ).dot( linalg.solve( self.data.A , self.data.H )) )
            invA_f = linalg.solve(self.data.A , self.data.outputs)
            invA_H = linalg.solve(self.data.A , self.data.H)

            longexp =\
            ( np.transpose(self.data.outputs) )\
            .dot(\
               invA_f - ( invA_H ).dot\
                  (\
                    linalg.solve( np.transpose(self.data.H).dot(invA_H) , np.transpose(self.data.H) )
                  )\
                 .dot( invA_f )\
                )

            ans = -0.5*(\
              -longexp - (np.log(signdetA)+logdetA) - np.log(val)\
              -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi)\
                       )
            
        #end = time.time()
        #print("time solver:" , end - start)

        ## calculate llh via cholesky decomposition - faster, more stable
        if True:
        #start = time.time()
        #for count in range(0,1000):
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))
            B = np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            longexp = ( np.transpose(self.data.outputs) )\
              .dot( invA_f - invA_H.dot(B) )

            #print(self.data.inputs[:,0].size)
            #print(self.data.inputs[0].size)

            ans = -0.5*\
              (-longexp - logdetA - np.log(linalg.det(Q))\
              -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi))\
        #end = time.time()
        #print("time cholesky:" , end - start)
        
        return ans
 
#        if signdetA > 0 and val > 0:
#            return ans
#        else:
#            print("ill conditioned covariance matrix...")
#            return 10000.0


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x):
        ## undo the transformation -- x is unscaled delta
        x = np.exp(x/2.0)

        ### calculate analytic sigma here ###
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.par.sigma=np.array([1.0])
        self.x_to_delta_and_sigma(np.append(x,self.par.sigma))
        self.data.make_A()

        if False:
        ## start time loop
        #start = time.time()
        #for count in range(0,1000):

            ## precompute terms depending on A^{-1}
            invA_f = linalg.solve(self.data.A , self.data.outputs)
            invA_H = linalg.solve(self.data.A , self.data.H)

            sig2 =\
              ( 1.0/(self.data.inputs[:,0].size - self.par.beta.size - 2.0) )\
                *( np.transpose(self.data.outputs) ).dot(\
                    invA_f - ( invA_H )\
                      .dot(\
                        np.linalg.solve(\
                          np.transpose(self.data.H).dot( invA_H ),\
                          np.transpose(self.data.H).dot( invA_f ) \
                        )\
                      )\
                )

            self.par.sigma = np.array([np.sqrt(sig2)])

            ### answers
            (signdetA, logdetA) = np.linalg.slogdet(self.data.A)
            #print("normal log:", np.log(signdetA)+logdetA)
     
            val=linalg.det( ( np.transpose(self.data.H) ).dot(\
              linalg.solve( self.data.A , self.data.H )) )

            ans = -(\
                        -0.5*(self.data.inputs[:,0].size - self.par.beta.size)\
                          *np.log( self.par.sigma[0]**2 )\
                        -0.5*(np.log(signdetA)+logdetA)\
                        -0.5*np.log(val)\
                       )
        ## end time loop
        #end = time.time()
        #print("time solvers:" , end - start)

        
        if True:
        ## start time loop
        #start = time.time()
        #for count in range(0,1000):
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))
            B = np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))

            #print(self.data.inputs[:,0].size , self.par.beta.size)

            sig2 =\
              ( 1.0/(self.data.inputs[:,0].size - self.par.beta.size - 2.0) )*\
                np.transpose(self.data.outputs).dot(invA_f-invA_H.dot(B)) \

            ## for MUCM, save sigma in parameters, not in kernel
            self.par.sigma = np.array([np.sqrt(sig2)])

            #logdetA = 2.0*np.trace(np.log(L))
            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            ans = -0.5*(\
                        -(self.data.inputs[:,0].size - self.par.beta.size)\
                          *np.log( self.par.sigma[0]**2 )\
                        -logdetA\
                        -np.log(np.linalg.det(Q))\
                      )
        ## end time loop
        #end = time.time()
        #print("time cholesky:" , end - start)

        return ans


    ## calculate sigma analytically - used for the MUCM method
    def sigma_analytic_mucm(self, x):
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.par.sigma=np.array([1.0])
        self.x_to_delta_and_sigma(np.append(x,self.par.sigma))
        self.data.make_A()

        ## stable numerical method
        L = np.linalg.cholesky(self.data.A) 
        w = np.linalg.solve(L,self.data.H)
        Q = w.T.dot(w)
        K = np.linalg.cholesky(Q)
        invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
        invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))
        B = np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))

        sig2 =\
          ( 1.0/(self.data.inputs[:,0].size - self.par.beta.size - 2.0) )*\
            np.transpose(self.data.outputs).dot(invA_f-invA_H.dot(B)) \

        ##  set sigma to its analytic value (but not in kernel)
        self.par.sigma = np.array([np.sqrt(sig2)])


    # calculates the optimal value of the mean hyperparameters
    def optimalbeta(self):
        #### fast - no direct inverses
        #invA_f = linalg.solve(self.data.A , self.data.outputs)
        #invA_H = linalg.solve(self.data.A , self.data.H)
        #self.par.beta = linalg.solve( np.transpose(self.data.H).dot(invA_H) , np.transpose(self.data.H) ).dot(invA_f)

        ## more stable
        L = np.linalg.cholesky(self.data.A) 
        w = np.linalg.solve(L,self.data.H)
        Q = w.T.dot(w)
        K = np.linalg.cholesky(Q)
        invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
        self.par.beta =\
          np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))


    # translate the loglikelihood function input 'x' back into delta and sigma
    def x_to_delta_and_sigma(self,x):
        x_read = 0
        x_temp = []
        for d in range(0, len(self.data.K.delta)):
            if self.data.K.delta[d].size > 0:
                d_per_dim = int(self.data.K.delta[d].flat[:].size/\
                  self.data.K.delta[d][0].size)
                x_temp.append(x[ x_read:x_read+self.data.K.delta[d].size ]\
                  .reshape(d_per_dim,self.data.K.delta[d][0].size))
            else:
                x_temp.append([])
            x_read = x_read + self.data.K.delta[d].size
        self.data.K.update_delta(x_temp)
 
        x_temp = []
        for s in range(0, len(self.data.K.sigma)):
            x_temp.append(x[ x_read:x_read+self.data.K.sigma[s].size ])
            #print(s, x_temp)
            x_read = x_read + self.data.K.sigma[s].size
        self.data.K.update_sigma(x_temp)
        #print("SIGMA:" , self.data.K.sigma)

        return
 
