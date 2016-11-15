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

        ## fixing hyperparameters
        self.fix = self.config.fix
 
        print("\n*** Optimization options ***")
 
        # if bounds are empty then construct them automatically
        if config.bounds == ():
            print("No bounds provided, so setting defaults based on data:")
            d_bounds_t = []
            s_bounds_t = []
            n_bounds_t = []

            # loop over the dimensions of the inputs for delta
            for i in range(0, self.data.inputs[0].size):
                data_range = np.amax(self.data.inputs[:,i]) - np.amin(self.data.inputs[:,i])
                print("    delta" , i , [0.001,data_range])
                d_bounds_t.append([0.001,data_range])

            # use small range for nugget
            data_range = np.sqrt( np.amax(self.data.outputs) - np.amin(self.data.outputs) )
            print("    nugget", [0.00001,0.0001])
            n_bounds_t.append([0.00000001,0.0000001])

            # use output range for sigma
            data_range = np.sqrt( np.amax(self.data.outputs) - np.amin(self.data.outputs) )
            print("    sigma", [0.001,data_range])
            s_bounds_t.append([0.001,data_range])

            ## BOUNDS
            config.bounds = tuple(d_bounds_t + n_bounds_t + s_bounds_t)
            # adjust bounds based on fix - the stored bounds won't include fixed parameters
            config.bounds = tuple(config.bounds[i] \
              for i in range(len(config.bounds)) if i not in self.fix)

            print("Data-based bounds:")
            print(config.bounds)
        else:
            config.bounds = tuple(config.bounds[i] \
              for i in range(len(config.bounds)) if i not in self.fix)
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

        i = 0 ## index of hyperparameter in x
        d_size = self.data.K.d.size
        for d in range(0, d_size):
            if d not in self.config.fix and i < len(bounds):

                hess = np.zeros(len(bounds)) ## this must only be for the active hyperparameters, and in this case not including the sigma... perhaps because all the delta are listed first it won't matter
                hess[i]=1.0
                dict_entry= {\
                            'type': 'ineq',\
                            'fun' : lambda x, f=i, lb=self.data.K.transform(0.001): x[f] - lb ,\
                            'jac' : lambda x, h=hess: h\
                            }
                self.cons.append(dict_entry)

                i = i + 1

        self.cons = tuple(self.cons)
        

    ## tries to keep within the specified bounds
    def bounds_constraint(self, bounds):
        print("setting up bounds constraint")
        self.cons = []
        # if i is in fixed we shouldn't make a constraint
        #for i in range(0, x_size - len(self.fix)):

        print("bounds:" , bounds)

        i = 0 ## index of hyperparameter in x
        x_size = self.data.K.d.size + 1 + 1
        for b in range(0, x_size):
            if b not in self.config.fix and i < len(bounds):

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

                i = i + 1

        MUCM = False
        if MUCM == True:
            del(self.cons[-1])

        self.cons = tuple(self.cons)
        return


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        use_cons = self.config.constraints
        bounds = self.config.bounds

        self.print_message = print_message

        print("Optimising delta and sigma...")

        ## scale the provided bounds
        bounds = self.transform(bounds)
        
        ## actual function containing the optimizer calls
        self.optimal(numguesses, use_cons, bounds)

        print("best hyperparameters: ")
        self.data.K.print_kernel()
        
        if self.beliefs.fix_mean == 'F':
            self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals = 4))

   
    def optimal(self, numguesses, use_cons, bounds):
        first_try = True
        best_min = 10000000.0

        #### how many parameters to fit ####

        ## params - number of paramaters that need fitting
        params = self.data.K.d.size + 1 + 1

        ## if MUCM case
        MUCM = False
        if MUCM == True:
            print("MUCM method: sigma is function of delta")
            params = params - 1 # no longer need to optimise sigma

        ## if fixed parameters
        params = params - len(self.fix)


        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## tell user which fitting method is being used
        if use_cons:
            print("Using COBYLA method (constaints)...")
        else:
            print("Using Nelder-Mead method (no constraints)...")

        ## sort out the fixed paramaters that we don't optimise
        fix = self.config.fix

        ## try each x-guess (start value for optimisation)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])

            ## constraints
            #if use_cons:
            if False:
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
                if self.print_message:
                    print(res, "\n")
            ## no constraints
            else:
                if MUCM:
                    res = minimize(self.loglikelihood_mucm,
                      x_guess, method = 'Nelder-Mead'\
                      ,options={'xtol':0.1, 'ftol':0.001})
                else:
                    #res = minimize(self.loglikelihood_gp4ml,
                    #  x_guess, method = 'Nelder-Mead'\
                    #  ,options={'xtol':0.1, 'ftol':0.001})
                    print("Trying this.")
                    res = minimize(self.loglikelihood_gp4ml,\
                      x_guess,jac=True,\
                        method='Newton-CG'\
                        )#, tol=0.1)
                if self.print_message:
                    print(res, "\n")
                    if res.success != True:
                        print(res.message, "Not succcessful.")
        
            ## result of fit
            print("  result: " , np.around(self.untransform(res.x),decimals=4),\
                  " llh: ", -1.0*np.around(res.fun,decimals=4))
            if (res.fun < best_min) or first_try:
                best_min = res.fun
                best_x = self.untransform(res.x)
                best_res = res
                first_try = False
        print("********")
        if MUCM:
            x = self.full_x(best_x , MUCM=True)
            self.sigma_analytic_mucm(x)  ## sets par.sigma correctly
            self.data.K.set_params(np.append(x , self.par.sigma))
        else:
            x = self.full_x(best_x)
            self.data.K.set_params(x)
        ## store these values in par, so we remember them
        self.par.delta = self.data.K.d
        self.par.sigma = self.data.K.s

        self.data.make_A()
        self.data.make_H()


    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x):
        ## undo the transformation...
        x = self.untransform(x)
        params = x.size

        ## reconstruct the "full x" as x doesn't include the fixed values 
        x = self.full_x(x)
        self.data.K.set_params(x) ## give values to kernels
        self.data.make_A() ## construct covariance matrix

        if self.print_message:
            self.data.K.print_kernel()

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

            ## calculate the gradients wrt hyperparameters
            grad_LLH = np.zeros(params)
            ### wrt delta
            for i in range(self.data.K.d.size):
                temp = self.data.K.grad_delta_A(self.data.inputs, i)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                grad_LLH[i] = -0.5* (\
                  np.trace(invA_gradHP) \
                  +np.transpose(self.data.outputs).dot(invA_gradHP).dot(invA_f) \
                                    )

            ### wrt nugget
            temp = self.data.K.grad_nugget_A(self.data.inputs)
            invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            grad_LLH[params - 2] = -0.5* (\
              np.trace(invA_gradHP) \
              +np.transpose(self.data.outputs).dot(invA_gradHP).dot(invA_f) \
                                )

            ### wrt sigma
            temp = self.data.K.grad_sigma_A(self.data.inputs)
            invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            grad_LLH[params-1] = -0.5* (\
              np.trace(invA_gradHP) \
              +np.transpose(self.data.outputs).dot(invA_gradHP).dot(invA_f) \
                                )

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
        
        return LLH , grad_LLH


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x):
        ## undo the transformation...
        x = self.untransform(x)

        ## reconstruct the "full x" as x doesn't include the fixed values 
        x = self.full_x(x, MUCM=True)

        ### calculate analytic sigma here ###
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.par.sigma=np.array([1.0])
        self.data.K.set_params(np.append(x,self.par.sigma))
        self.data.make_A()

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

            self.par.sigma = np.array([np.sqrt(sig2)])

            ### answers
            (signdetA, logdetA) = np.linalg.slogdet(self.data.A)
            #print("normal log:", np.log(signdetA)+logdetA)
     
            val=linalg.det( ( np.transpose(self.data.H) ).dot(\
              linalg.solve( self.data.A , self.data.H )) )

            if signdetA > 0 and val > 0:
                ans = -(\
                            -0.5*(self.data.inputs[:,0].size - self.par.beta.size)\
                              *np.log( self.par.sigma[0]**2 )\
                            -0.5*(np.log(signdetA)+logdetA)\
                            -0.5*np.log(val)\
                           )
            else:
                print("Ill conditioned covariance matrix... try using nugget.")
                ans = 10000.0
                exit()

            #end = time.time()
            #print("time solvers:" , end - start)

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


    ## return list of all hyperparameters (both fitting ones and fixed ones)
    def full_x(self, x, MUCM = False):
        fix = self.config.fix

        x_all = self.data.K.fetch_params()

        params = len(x_all)
        if MUCM:
            x_all = x_all[:-1]
            params = params - 1

        j = 0
        for i in range(0, params):
            if i not in fix:
                x_all[i] = x[j]
                j = j + 1

        return np.array(x_all)


    def transform(self, x):
        return 2.0*np.log(x)

    def untransform(self, x):
        return np.exp(x/2.0)

