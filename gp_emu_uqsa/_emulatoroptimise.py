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
        d_bounds_t = []
        n_bounds_t = []
        s_bounds_t = []
 
        if config.delta_bounds == []:
            print("Data-based bounds for delta:")
            # loop over the dimensions of the inputs for delta
            for i in range(0, self.data.inputs[0].size):
                data_range = np.amax(self.data.inputs[:,i])\
                           - np.amin(self.data.inputs[:,i])
                d_bounds_t.append([0.001,data_range])
                print("    delta" , i , '[{:04.3f} , {:04.3f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]) )
        else:
            print("User provided bounds for delta:")

            if len(config.delta_bounds) != self.data.inputs[0].size:
                print("ERROR: Wrong number of delta_bounds specified, exiting.")
                exit()

            for i in range(0, self.data.inputs[0].size):
                if config.delta_bounds[i] == []:
                    data_range = np.amax(self.data.inputs[:,i])\
                               - np.amin(self.data.inputs[:,i])
                    d_bounds_t.append([0.001,data_range])
                    print("    delta" , i , '[{:04.3f} , {:04.3f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]), "(data)")
                else:
                    d_bounds_t.append(config.delta_bounds[i])
                    print("    delta" , i , '[{:04.3f} , {:04.3f}]'.format(d_bounds_t[i][0] , d_bounds_t[i][1]), "(user)")

            #for i in range(0, self.data.inputs[0].size):
            #    print("    delta" , i , d_bounds_t[i])

        if config.nugget_bounds == []:
            print("Data-based bounds for nugget:")
            # use small range for nugget
            data_range = np.sqrt( np.amax(self.data.outputs)\
                       - np.amin(self.data.outputs) )
            n_bounds_t.append([0.0001,0.01])
            print("    nugget" , '[{:04.3f} , {:04.3f}]'.format(n_bounds_t[0][0] , n_bounds_t[0][1]) )
        else:
            print("User provided bounds for nugget:")
            n_bounds_t = config.nugget_bounds
            print("    nugget" , '[{:04.3f} , {:04.3f}]'.format(n_bounds_t[0][0] , n_bounds_t[0][1]) )

        if config.sigma_bounds == []:
            print("Data-based bounds for sigma:")
            # use output range for sigma
            data_range = np.sqrt( np.amax(self.data.outputs)\
                       - np.amin(self.data.outputs) )
            s_bounds_t.append([0.001,data_range])
            print("    sigma" , '[{:04.3f} , {:04.3f}]'.format(s_bounds_t[0][0] , s_bounds_t[0][1]) )
        else:
            print("User provided bounds for sigma:")
            s_bounds_t = config.sigma_bounds
            print("    sigma" , '[{:04.3f} , {:04.3f}]'.format(s_bounds_t[0][0] , s_bounds_t[0][1]) )

        ## different bounds depending on scenario
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T':
                config.bounds = tuple(d_bounds_t + n_bounds_t)
            else:
                config.bounds = tuple(d_bounds_t + n_bounds_t + s_bounds_t)
        else:
            if self.beliefs.mucm == 'T':
                config.bounds = tuple(d_bounds_t)
            else:
                config.bounds = tuple(d_bounds_t + s_bounds_t)
        #print("bounds:" , config.bounds)

        # set up type of bounds
        if config.constraints == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            self.standard_constraint(config.bounds)

        
    ## tries to keep deltas above a small value
    def standard_constraint(self, bounds):
        print("Setting up standard constraint")
        self.cons = []

        d_size = self.data.K.d.size
        for i in range(0, d_size):
            self.cons.append([self.data.K.transform(0.001),None])
         
        if self.beliefs.fix_nugget == 'F':
            self.cons.append([None,None])
        if self.beliefs.mucm == 'F':
            self.cons.append([None,None])

        return 


    ## tries to keep within the specified bounds
    def bounds_constraint(self, bounds):
        print("Setting up bounds constraint")
        self.cons = []

        x_size = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T':
                x_size = x_size + 1
            else:
                x_size = x_size + 2
        else:
            if self.beliefs.mucm == 'T':
                x_size = x_size
            else:
                x_size = x_size + 1

        for i in range(0, x_size):
            lower, upper = bounds[i]
            self.cons.append([self.data.K.transform(lower),\
                              self.data.K.transform(upper)])

        return


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        bounds = self.config.bounds

        self.print_message = print_message

        print("Optimising hyperparameters...")

        ## transform the provided bounds
        bounds = self.data.K.transform(bounds)
        #print(bounds)       
 
        ## actual function containing the optimizer calls
        self.optimal(numguesses, bounds)

        print("best hyperparameters: ")
        self.data.K.print_kernel()
        print("sigma:" , self.par.sigma)

        if self.beliefs.fix_nugget == 'F':
            noisesig = np.sqrt(self.par.sigma**2 * (self.par.nugget)/(1.0-self.par.nugget))
            print("'noise sigma' estimate from nugget:" , noisesig)
            
        
        self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals = 4))

   
    def optimal(self, numguesses, bounds):
        first_try = True
        best_min = 10000000.0

        ## params - number of paramaters that need fitting
        params = self.data.K.d.size
        if self.beliefs.fix_nugget == 'F':
            if self.beliefs.mucm == 'T':
                params = params + 1
            else:
                params = params + 2
        else:
            if self.beliefs.mucm == 'T':
                params = params
            else:
                params = params + 1
        
        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## print information about which parameters we're fitting
        if self.beliefs.fix_nugget == 'F':
            print("Training nugget on data")

        if self.beliefs.mucm == 'T':
            print("Using MUCM method for sigma")

        ## tell user which fitting method is being used
        if self.config.constraints != "none":
            print("Using L-BFGS-G method (with constraints)...")
        else:
            print("Using L-BFGS-G method (no constraints)...")

        ## try each x-guess (start value for optimisation)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])

            ## constraints - must use bounds for L-BFGS-B method
            if self.config.constraints != "none":

                if self.beliefs.mucm == 'T':
                    res = minimize(self.loglikelihood_mucm,
                      x_guess, method = 'L-BFGS-B', jac=True, bounds=self.cons)
                else:
                    res = minimize(self.loglikelihood_gp4ml,
                      x_guess, method = 'L-BFGS-B', jac=True, bounds=self.cons)

            ## no constraints
            else:
                if self.beliefs.mucm == 'T':
                    res = minimize(self.loglikelihood_mucm,
                      x_guess, method = 'L-BFGS-B', jac=True)
                else:
                    res = minimize(self.loglikelihood_gp4ml,
                      x_guess, method = 'L-BFGS-B', jac=True)

            if self.print_message:
                print(res, "\n")
                if res.success != True:
                    print(res.message, "Not succcessful.")
        
            ## result of fit
            sig_str = "" 
            if self.beliefs.mucm == 'T':
                self.sigma_analytic_mucm(self.data.K.untransform(res.x))
                sig_str = "  sig: " + str(np.around(self.par.sigma,decimals=4))
            print("  hp: ",\
                np.around(self.data.K.untransform(res.x),decimals=4),\
                " llh: ", -1.0*np.around(res.fun,decimals=4) , sig_str)
                
            ## set best result
            if (res.fun < best_min) or first_try:
                best_min = res.fun
                best_x = self.data.K.untransform(res.x)
                best_res = res
                first_try = False

        print("********")
        if self.beliefs.mucm == 'T':
            self.data.K.set_params(best_x)
            self.par.delta = self.data.K.d
            self.par.nugget = self.data.K.n
            self.sigma_analytic_mucm(best_x)
        else:
            self.data.K.set_params(best_x[:-1])
            self.par.delta = self.data.K.d
            self.par.nugget = self.data.K.n
            self.par.sigma = best_x[-1]

        self.data.make_A()
        self.data.make_H()


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x)
        self.data.make_A()

        try:
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))

            solve_K_HT = np.linalg.solve(K,self.data.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_f))

            invA_H_dot_B = invA_H.dot(B)
            sig2 =\
              ( 1.0/(self.data.inputs[:,0].size - self.par.beta.size - 2.0) )*\
                np.transpose(self.data.outputs).dot(invA_f-invA_H_dot_B)

            self.par.sigma = np.sqrt(sig2)

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            LLH = -0.5*(\
                        -(self.data.inputs[:,0].size - self.par.beta.size)\
                          *np.log( self.par.sigma**2 )\
                        -logdetA\
                        -np.log(np.linalg.det(Q))\
                       )

            ## calculate the gradients wrt hyperparameters
            grad_LLH = np.empty(x.size)
            
            H_dot_B = self.data.H.dot(B).T
           
            factor = (self.data.inputs[:,0].size - self.par.beta.size)/ \
                     (sig2*(self.data.inputs[:,0].size - self.par.beta.size - 2))

            #### wrt delta
            for i in range(self.data.K.d.size):
                temp = self.data.K.grad_delta_A(self.data.inputs[:,i], i, sig2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[i] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  + factor * ( \
                    +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                    +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                             ) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                    )

            #### wrt nugget
            if x.size == self.data.K.d.size + 1: ## if x contains nugget value
                temp = self.data.K.grad_nugget_A(self.data.inputs, sig2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[x.size-1] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  + factor * ( \
                    +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                    +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                             ) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                           )

        except np.linalg.linalg.LinAlgError as e:
            print("In loglikelihood_mucm(), matrix not PSD,"
                  " try nugget (or adjust nugget bounds).")
            exit()

        return LLH, grad_LLH


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm1(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x)
        self.data.make_A()

        try:
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

        except np.linalg.linalg.LinAlgError as e:
            print("In loglikelihood_mucm(), matrix not PSD,"
                  " try nugget (or adjust nugget bounds).")
            exit()

        return LLH

    ## calculate sigma analytically - used for the MUCM method
    def sigma_analytic_mucm(self, x):
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.data.K.set_params(x)
        self.data.make_A()

        try:
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

        except np.linalg.linalg.LinAlgError as e:
            print("In sigma_analytic_mucm(), matrix not PSD,"
                  " try nugget (or adjust nugget bounds).")
            exit()

        return


    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x[:-1]) # not including sigma in x
        self.data.make_A()

        ## for now, let's just multiply A by sigma**2
        self.par.sigma = x[-1]
        s2 = x[-1]**2
        self.data.A = s2*self.data.A

        try:
            L = np.linalg.cholesky(self.data.A) 
            w = np.linalg.solve(L,self.data.H)
            Q = w.T.dot(w)
            K = np.linalg.cholesky(Q)
            invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
            invA_H = np.linalg.solve(L.T, np.linalg.solve(L,self.data.H))

            solve_K_HT = np.linalg.solve(K,self.data.H.T)
            B = np.linalg.solve(K.T, solve_K_HT.dot(invA_f))

            logdetA = 2.0*np.sum(np.log(np.diag(L)))

            invA_H_dot_B = invA_H.dot(B)
            longexp = ( self.data.outputs.T ).dot( invA_f - invA_H_dot_B )

            LLH = -0.5*\
              (-longexp - logdetA - np.log(linalg.det(Q))\
               -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi))

            ## calculate the gradients wrt hyperparameters
            grad_LLH = np.empty(x.size)
            
            H_dot_B = self.data.H.dot(B).T
            
            #### wrt delta
            for i in range(self.data.K.d.size):
                temp = self.data.K.grad_delta_A(self.data.inputs[:,i], i, s2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[i] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                  +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                    )

            #### wrt nugget
            if x.size == self.data.K.d.size + 2: ## if x contains nugget value
                temp = self.data.K.grad_nugget_A(self.data.inputs, s2)
                invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
                sam = (invA_gradHP).dot(invA_H_dot_B)
                grad_LLH[x.size-2] = -0.5* (\
                  -np.trace(invA_gradHP) \
                  +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
                  +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
                  + np.trace( np.linalg.solve(K.T, \
                                solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                           )

            #### wrt sigma ## in gp4ml LLH sigma is always in x
            temp = self.data.A ## already X s2
            invA_gradHP = np.linalg.solve(L.T, np.linalg.solve(L,temp))
            sam = (invA_gradHP).dot(invA_H_dot_B)
            grad_LLH[x.size-1] = -0.5* (\
              -np.trace(invA_gradHP) \
              +(self.data.outputs.T).dot(invA_gradHP).dot(invA_f) \
              +( - 2*self.data.outputs.T + H_dot_B ).dot(sam) \
              + np.trace( np.linalg.solve(K.T, \
                            solve_K_HT.dot(invA_gradHP)).dot(invA_H) )
                                )

        except np.linalg.linalg.LinAlgError as e:
            print("In loglikelihood_gp4ml(), matrix not PSD,"
                  " try nugget (or adjust nugget bounds).")
            exit()

        return LLH, grad_LLH

    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml1(self, x):
        x = self.data.K.untransform(x)
        self.data.K.set_params(x[:-1]) # not including sigma in x
        self.data.make_A()

        ## for now, let's just multiply A by sigma**2
        self.par.sigma = x[-1]
        s2 = x[-1]**2
        self.data.A = s2*self.data.A

        try:
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

            LLH = -0.5*\
              (-longexp - logdetA - np.log(linalg.det(Q))\
              -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi))

        except np.linalg.linalg.LinAlgError as e:
            print("In loglikelihood_gp4ml(), matrix not PSD,"
                  " try nugget (or adjust nugget bounds).")
            exit()

        return LLH

    # calculates the optimal value of the mean hyperparameters
    def optimalbeta(self):
        L = np.linalg.cholesky(self.data.A) 
        w = np.linalg.solve(L,self.data.H)
        Q = w.T.dot(w)
        K = np.linalg.cholesky(Q)
        invA_f = np.linalg.solve(L.T, np.linalg.solve(L,self.data.outputs))
        self.par.beta =\
          np.linalg.solve(K.T, np.linalg.solve(K,self.data.H.T).dot(invA_f))


