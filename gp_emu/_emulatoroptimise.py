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
        self.standard_constraint()
       
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

            # loop over kernels
            #print("Setting delta bounds...")
            for k in range(0, len(self.data.K.name)):
                print(self.data.K.name[k])

                # loop over different delta within a kernel
                for dn in self.data.K.delta_names[k]:
                    print(" " , dn)

                    # loop over the dimensions of the inputs
                    for i in range(0, self.data.inputs[0].size):
                        data_range = np.amax(self.data.inputs[:,i]) - np.amin(self.data.inputs[:,i])
                        print("    dim" , i , [0.001,data_range])
                        d_bounds_t.append([0.001,data_range])

                # loop over different sigma within a kernel
                for sn in self.data.K.sigma_names[k]:
                    data_range = np.sqrt( np.amax(self.data.outputs) - np.amin(self.data.outputs) )
                    print(" " , sn , [0.001,data_range])
                    s_bounds_t.append([0.001,data_range])

            config.bounds = tuple(d_bounds_t + s_bounds_t)

            # adjust bounds based on fix
            config.bounds = tuple(config.bounds[i] \
              for i in range(len(config.bounds)) if i not in self.fix)

            #print("Data-based bounds:")
            #print(config.bounds)
        else:
            print("User provided bounds:")
            #print(config.bounds)

            # adjust bounds based on fix
            config.bounds = tuple(config.bounds[i] \
              for i in range(len(config.bounds)) if i not in self.fix)

            dub = 0
            sub = 0
            # loop over kernels
            #print("Delta bounds...")
            x_num = 0
            for k in range(0, len(self.data.K.name)):
                print(self.data.K.name[k])

                # loop over different delta within a kernel
                for dn in self.data.K.delta_names[k]:
                    print(" " , dn)

                    # loop over the dimensions of the inputs
                    for i in range(0, self.data.inputs[0].size):
                        #if x_num not in self.fix:
                        print("    dim" , i , config.delta_bounds[dub])
                        #else:
                        #    print("    dim" , i , "is fixed, so bounds not used")
                        dub = dub + 1
                        x_num = x_num + 1
            
                # loop over different sigma within a kernel
                for sn in self.data.K.sigma_names[k]:
                    #if x_num not in self.fix:
                    print(" " , sn , config.sigma_bounds[sub])
                    #else:
                    #    print("    fixed, so no bounds used")
                    sub = sub + 1
                    x_num = x_num + 1


        # set up type of bounds
        if config.constraints_type == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            self.standard_constraint()
        
    ## tries to keep deltas above a small value
    def standard_constraint(self):
        print("setting up standard constraint")
        self.cons = []
        #for i in range(0, self.data.K.delta_num):
        for i in range(0, self.data.K.delta_num):
            # if i is in fixed we shouldn't make a constraint
            if i not in self.config.fix:
                hess = np.zeros(self.data.K.delta_num)
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
        print("setting up bounds constraint")
        self.cons = []
        x_size = self.data.K.delta_num + self.data.K.sigma_num
        # in case of MUCM llh, not fitting sigma 
        if len(self.data.K.name)==1 and self.data.K.name[0]=="gaussian_mucm":
            x_size = x_size - 1
        # if i is in fixed we shouldn't make a constraint
        #for i in range(0, x_size - len(self.fix)):
        for i in range(0, len(bounds)):
            hess = np.zeros(x_size)
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


    def llh_optimize(self, print_message=False):

        numguesses = self.config.tries
        use_cons = self.config.constraints
        bounds = self.config.bounds
        stochastic = self.config.stochastic

        self.print_message = print_message

        print("Optimising delta and sigma...")

        ### scale the provided bounds
        bounds_new = []
        for i in bounds:
            temp = 2.0*np.log(np.array(i))
            bounds_new = bounds_new + [list(temp)]
        bounds = tuple(bounds_new)
        
        ## actual function containing the optimizer calls
        self.optimal(numguesses, use_cons, bounds, stochastic)

        #print("best delta: " , self.par.delta)
        #print("best sigma: " , self.par.sigma)
        print("best hyperparameters: ")
        self.data.K.print_kernel()
        
        #print("best sigma**2: ", [[j**2 for j in i] for i in self.par.sigma])

        if self.beliefs.fix_mean == 'F':
            self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals = 4))

   
    def optimal(self,\
      numguesses, use_cons, bounds, stochastic):
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

        params = params - len(self.fix)
        ## construct list of guesses from bounds
        guessgrid = np.zeros([params, numguesses])
        print("Calculating initial guesses from bounds")
        for R in range(0, params):
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ## tell user which fitting method is being used
        if stochastic:
            print("Using global stochastic method (bounded)...")
        else:
            if use_cons:
                print("Using COBYLA method (constaints)...")
            else:
                print("Using Nelder-Mead method (no constraints)...")

        ## sort out the fixed paramaters that we don't optimise
        fix = self.config.fix

        ## try each x-guess (start value for optimisation)
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])
            #x_guess = [guessgrid[:,C][i] for i in range(0,len(guessgrid[:,C])) if i not in fix]
            #bounds = tuple(bounds[i] for i in range(0, len(bounds)) if i not in fix)
            #print("x_guess:" , x_guess)
            #print("bounds:" , bounds)
            #print("constraints:")
            #for i in self.cons:
            #    print(i)

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
                        if self.print_message:
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
                            #print("print kernel in optimal() function")
                            #self.data.K.print_kernel()
                            res = minimize(self.loglikelihood_gp4ml,\
                              x_guess,constraints=self.cons,\
                                method='COBYLA'\
                                )#, tol=0.1)
                        if self.print_message:
                            print(res, "\n")
                    else:
                        if MUCM:
                            res = minimize(self.loglikelihood_mucm,
                              x_guess, method = 'Nelder-Mead'\
                              ,options={'xtol':0.1, 'ftol':0.001})
                        else:
                            res = minimize(self.loglikelihood_gp4ml,
                              x_guess, method = 'Nelder-Mead'\
                              ,options={'xtol':0.1, 'ftol':0.001})
                        if self.print_message:
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
            x = self.full_x(best_x)
            self.x_to_delta_and_sigma(x)
        ## store these values in par, so we remember them
        self.par.delta = [[list(i) for i in d] for d in self.data.K.delta]
        self.par.sigma = [list(s) for s in self.data.K.sigma]

        self.data.make_A()
        self.data.make_H()


    # the loglikelihood provided by Gaussian Processes for Machine Learning 
    def loglikelihood_gp4ml(self, x):
        x = np.exp(x/2.0) ## undo the transformation...

        ## reconstruct the "full x" as x doesn't include the fixed values 
        #print("before")
        x = self.full_x(x)
        #self.delta_and_sigma_to_x()

        self.x_to_delta_and_sigma(x) ## give values to kernels
        self.data.make_A() ## construct covariance matrix

        #print("after")
        #x = self.full_x(x)

        #print("print kernel in loglikelihood_gp4ml() function")
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

            ans = -0.5*\
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
                ans = -0.5*(\
                  -longexp - (np.log(signdetA)+logdetA) - np.log(val)\
                  -(self.data.inputs[:,0].size-self.par.beta.size)*np.log(2.0*np.pi) )
            else:
                print("Ill conditioned covariance matrix... try using nugget.")
                ans = 10000.0
                exit()
            #end = time.time()
            #print("time solver:" , end - start)
        
        return ans


    # the loglikelihood provided by MUCM
    def loglikelihood_mucm(self, x):
        ## undo the transformation -- x is unscaled delta
        x = np.exp(x/2.0)

        ### calculate analytic sigma here ###
        ## to match my covariance matrix to the MUCM matrix 'A'
        self.par.sigma=np.array([1.0])
        self.x_to_delta_and_sigma(np.append(x,self.par.sigma))
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
    def full_x(self, x):
        fix = self.config.fix

        x_all = self.delta_and_sigma_to_x()

        #print("x:" , x)
        #print("x_all:" , x_all)

        j = 0
        for i in range(0, len(x_all)):
            if i not in fix:
                x_all[i] = x[j]
                j = j + 1

        return np.array(x_all)

    
    # translate delta and sigma into x
    def delta_and_sigma_to_x(self):
        x_read = 0

        x_all = []
        # loop over kernel
        for k in range(0, len(self.data.K.name)):
            d_size_k = self.data.K.delta[k].size
            if d_size_k > 0:
                #print("delta[k]:" , self.data.K.delta[k])
                #print("delta[k].flatten():" , self.data.K.delta[k].flatten())
                #print("list(delta[k].flatten()):" , list(self.data.K.delta[k].flatten()))
                x_all = x_all + list(self.data.K.delta[k].flatten())
            x_read = x_read + d_size_k
        #print("x_all:" , x_all)
 
        # loop over kernel
        for k in range(0, len(self.data.K.sigma)):
            s_size_k = self.data.K.sigma[k].size
            #print("sigma[k]:" , self.data.K.sigma[k])
            #print("sigma[k].flatten():" , self.data.K.sigma[k].flatten())
            #print("list(sigma[k].flatten()):" , list(self.data.K.sigma[k].flatten()))
            x_all = x_all + list(self.data.K.sigma[k].flatten())
            x_read = x_read + s_size_k

        return x_all

    # translate the loglikelihood function input 'x' back into delta and sigma
    def x_to_delta_and_sigma(self,x):
        x_read = 0

        x_temp = []
        # loop over kernel
        for k in range(0, len(self.data.K.name)):
        #for d in range(0, len(self.data.K.delta)):

            # number of delta in this kernel
            d_size_k = self.data.K.delta[k].size
            if d_size_k > 0:
                # number of delta per input dimension
                d_per_dim = self.data.K.delta[k].shape[0]
                #d_per_dim = int(self.data.K.delta[k].flat[:].size/\
                #  self.data.K.delta[k][0].size)
                x_temp.append(x[ x_read:x_read+d_size_k ].reshape(d_per_dim,int(d_size_k/d_per_dim)))
                  #.reshape(d_per_dim, self.data.K.delta[k][0].size))
            else:
                x_temp.append([])
            x_read = x_read + d_size_k
            #x_read = x_read + self.data.K.delta[k].size
        #print("x_temp:" , x_temp)
        self.data.K.update_delta(x_temp)
 
        x_temp = []
        # loop over kernel
        for k in range(0, len(self.data.K.sigma)):

            # number of delta in this kernel
            s_size_k = self.data.K.sigma[k].size

            x_temp.append(x[ x_read:x_read + s_size_k ])
            #print(s, x_temp)
            x_read = x_read + s_size_k
            #x_read = x_read + self.data.K.sigma[k].size
        self.data.K.update_sigma(x_temp)
        #print("SIGMA:" , self.data.K.sigma)

        #print("print kernel in x_to_delta_and_sigma() function")
        #self.data.K.print_kernel()
        return
 
