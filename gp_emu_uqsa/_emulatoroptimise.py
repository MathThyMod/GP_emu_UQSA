from __future__ import print_function
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import time

### for optimising the hyperparameters
class Optimize:
    def __init__(self, data, basis, par, beliefs, config):
        self.data = data
        self.basis = basis
        self.par = par
        self.beliefs = beliefs
        self.config = config
       
        ## for message printing - off by default
        self.print_message = False

        print("\n*** Optimization options ***")
 
        # if bounds are empty then construct them automatically
        if config.bounds == ():
            print("No bounds provided, so setting defaults based on data:")
            d_bounds_t = []
            n_bounds_t = []

            # loop over the dimensions of the inputs for delta
            for i in range(0, self.data.inputs[0].size):
                data_range = np.amax(self.data.inputs[:,i]) - np.amin(self.data.inputs[:,i])
                print("    delta" , i , [0.001,data_range])
                d_bounds_t.append([0.001,data_range])

            # use small range for nugget
            data_range = np.sqrt( np.amax(self.data.outputs) - np.amin(self.data.outputs) )
            print("    nugget", [0.0001,0.01])
            n_bounds_t.append([0.0001,0.01])

            # use output range for sigma
            data_range = np.sqrt( np.amax(self.data.outputs) - np.amin(self.data.outputs) )
            print(" " , sn , [0.001,data_range])
            s_bounds_t.append([0.001,data_range])

            ## BOUNDS
            if self.beliefs.fix_nugget == 'F':
                config.bounds = tuple(d_bounds_t + n_bounds_t + s_bounds_t)
            else:
                config.bounds = tuple(d_bounds_t)

            print("Data-based bounds:")
            print(config.bounds)
        else:
            print("User provided bounds:")
            print(config.bounds)


        # set up type of bounds
        if config.constraints == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            self.standard_constraint(config.bounds)

        
    ## tries to keep deltas above a small value
    def standard_constraint(self, bounds):
        print("setting up standard constraint")
        self.cons = []

        d_size = self.data.K.d.size
        for i in range(0, d_size):

            hess = np.zeros(len(bounds))
            hess[i]=1.0
            dict_entry= {\
                        'type': 'ineq',\
                        'fun' : lambda x, f=i, lb=self.data.K.transform(0.001): x[f] - lb ,\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)

        self.cons = tuple(self.cons)
        

    ## tries to keep within the specified bounds
    def bounds_constraint(self, bounds):
        print("setting up bounds constraint")
        self.cons = []

        x_size = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            x_size = x_size + 2

        for i in range(0, x_size):

            hess = np.zeros(len(bounds))
            hess[i]=1.0
            lower, upper = bounds[i]

            dict_entry = {\
              'type': 'ineq',\
              'fun' : lambda x, f=i, lb=self.data.K.transform(lower): x[f] - lb ,\
              'jac' : lambda x, h=hess: h\
            }
            self.cons.append(dict_entry)
            dict_entry = {\
              'type': 'ineq',\
              'fun' : lambda x, f=i, ub=self.data.K.transform(upper): ub - x[f] ,\
              'jac' : lambda x, h=hess: h\
            }
            self.cons.append(dict_entry)

        self.cons = tuple(self.cons)
        return


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        bounds = self.config.bounds

        self.print_message = print_message

        print("Optimising delta and sigma...")

        ## transform the provided bounds
        bounds = self.data.K.transform(bounds)
        print(bounds)       
 
        ## actual function containing the optimizer calls
        self.optimal(numguesses, bounds)

        print("best hyperparameters: ")
        self.data.K.print_kernel()
        print("sigma:" , self.par.sigma)
        
        if self.beliefs.fix_mean == 'F':
            self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals = 4))

   
    def optimal(self, numguesses, bounds):
        first_try = True
        best_min = 10000000.0

        ## params - number of paramaters that need fitting
        params = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            params = params + 2

        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## tell user which fitting method is being used
        if self.config.constraints != "none":
            print("Using COBYLA method (constaints)...")
        else:
            print("Using Nelder-Mead method (no constraints)...")

        ## try each x-guess (start value for optimisation)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])

            ## constraints
            if self.config.constraints != "none":

                if self.beliefs.fix_nugget == 'F':
                    print("training nugget, so training sigma too")
                    res = minimize(self.loglikelihood_gp4ml,\
                      x_guess,constraints=self.cons,\
                        method='COBYLA'\
                        )#, tol=0.1)
                else:
                    print("not training nugget, so sigma is analytic")
                    res = minimize(self.loglikelihood_mucm,\
                      x_guess,constraints=self.cons,\
                        method='COBYLA'\
                        )#, tol=0.1)
                if self.print_message:
                    print(res, "\n")

            ## no constraints
            else:
                res = minimize(self.loglikelihood_mucm,
                  x_guess, method = 'Nelder-Mead'\
                  ,options={'xtol':0.1, 'ftol':0.001})
                #res = minimize(self.loglikelihood_mucm,\
                #  x_guess,jac=True,\
                #    method='Newton-CG')

                if self.print_message:
                    print(res, "\n")
                    if res.success != True:
                        print(res.message, "Not succcessful.")
        
            ## result of fit
            print("  hp: ",\
                np.around(self.data.K.untransform(res.x),decimals=4),\
                " llh: ", -1.0*np.around(res.fun,decimals=4))
            ## set best result
            if (res.fun < best_min) or first_try:
                best_min = res.fun
                best_x = self.data.K.untransform(res.x)
                best_res = res
                first_try = False

        print("********")
        if self.beliefs.fix_nugget == 'F':
            self.data.K.set_params(best_x[:-1])
            self.par.delta = self.data.K.d
            self.par.nugget = self.data.K.n
            self.par.sigma = best_x[-1]  ## sets par.sigma correctly
        else:
            self.data.K.set_params(best_x)
            self.par.delta = self.data.K.d
            self.par.nugget = self.data.K.n
            self.sigma_analytic_mucm(best_x)  ## sets par.sigma correctly

        self.data.make_A()
        self.data.make_H()


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x)
        self.data.make_A()

        #self.data.K.print_kernel()

        try:
        #start = time.time()
        #for count in range(0,1000):

            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))
            B = np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))

            sig2 =\
              ( 1.0/(self.data.inputs[:,0].size - self.par.beta.size - 2.0) )*\
                np.transpose(self.data.outputs).dot(invA_f-invA_H.dot(B))

            self.par.sigma = np.sqrt(sig2)

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            LLH = -0.5*(\
                        -(self.data.inputs[:,0].size - self.par.beta.size)\
                          *np.log( self.par.sigma**2 )\
                        -logdetA\
                        -np.log(np.linalg.det(Q))\
                       )

            ## calculate the gradients wrt hyperparameters
            #grad_LLH = np.zeros(params)
            #### wrt delta
            #for i in range(self.data.K.d.size):
            #    temp = self.data.K.grad_delta_A(self.data.inputs, i)
            #    invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            #    grad_LLH[i] = -0.5* (\
            #      np.trace(invA_gradHP) \
            #      +np.transpose(self.data.outputs).dot(invA_gradHP).dot(invA_f) \
            #                        )

            #### wrt nugget
            #temp = self.data.K.grad_nugget_A(self.data.inputs)
            #invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            #grad_LLH[params - 2] = -0.5* (\
            #  np.trace(invA_gradHP) \
            #  +np.transpose(self.data.outputs).dot(invA_gradHP).dot(invA_f) \
            #                    )

        #end = time.time()
        #print("time cholesky:" , end - start)

        except np.linalg.linalg.LinAlgError as e:
            print("Matrix not PSD, trying direct solve instead of Cholesky decomp.")    
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

            self.par.sigma = np.sqrt(sig2)

            ### LLHwers
            (signdetA, logdetA) = np.linalg.slogdet(self.data.A)
            #print("normal log:", np.log(signdetA)+logdetA)
     
            val=linalg.det( ( np.transpose(self.data.H) ).dot(\
              linalg.solve( self.data.A , self.data.H )) )

            if signdetA > 0 and val > 0:
                LLH = -(\
                            -0.5*(self.data.inputs[:,0].size - self.par.beta.size)\
                              *np.log( self.par.sigma**2 )\
                            -0.5*(np.log(signdetA)+logdetA)\
                            -0.5*np.log(val)\
                           )
            else:
                print("Ill conditioned covariance matrix... try using nugget.")
                LLH = 10000.0
                exit()

            #end = time.time()
            #print("time solvers:" , end - start)

        return LLH


    ## calculate sigma analytically - used for the MUCM method
    def sigma_analytic_mucm(self, x):
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.data.K.set_params(x)
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
            np.transpose(self.data.outputs).dot(invA_f-invA_H.dot(B))

        ##  set sigma to its analytic value (but not in kernel)
        self.par.sigma = np.sqrt(sig2)


    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x[:-1]) # not including sigma in x
        self.data.make_A()

        #self.data.K.print_kernel()

        ## for now, let's just multiply A by sigma**2
        self.par.sigma = x[-1]
        s2 = x[-1]**2
        self.data.A = s2*self.data.A

        ## calculate llh via cholesky decomposition - faster, more stable
        try:
        #start = time.time()
        #for count in range(0,10):

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

            LLH = -0.5*\
              (-longexp - logdetA - np.log(linalg.det(Q))\
              -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi))

        #end = time.time()
        #print("time cholesky:" , end - start)

        except np.linalg.linalg.LinAlgError as e:
            print("Matrix not PSD, trying direct solve instead of Cholesky decomp.")    
            ## calculate llh via solver routines - slower, less stable

            #start = time.time()
            #for count in range(0,10):

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

            if signdetA > 0 and val > 0:
                LLH = -0.5*(\
                  -longexp - (np.log(signdetA)+logdetA) - np.log(val)\
                  -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi) )
            else:
                print("Ill conditioned covariance matrix... try using nugget.")
                LLH = 10000.0
                exit()
            #end = time.time()
            #print("time solver:" , end - start)
        
        return LLH


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


