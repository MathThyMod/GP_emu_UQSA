from __future__ import print_function
import numpy as _np
import scipy.spatial.distance as _dist

_np.set_printoptions(precision=6)
_np.set_printoptions(suppress=True)

########################
### kernel functions ###
########################

### use kernel specification in Beliefs to build the kernel
def build_kernel(beliefs):
    print("\n*** Building kernel ***")
    n = 0
    k_com = ""
    for i in beliefs.kernel:
        if n == 0:
            k_com = i
        if n > 0:
            k_com = k_com + " + " + i
        n = n + 1

    K = eval(k_com)

    return K
    

### configure kernel with enough delta for all the subkernels 
def auto_configure_kernel(K, par, all_data):

    # construct list of required delta, because number depends on input dim
    d_list = [ ]
    for d in range(0, len(K.name)):
        if K.delta[d].size != 0:
            d_per_dim = int(K.delta[d].flat[:].size / K.delta[d][0].size)
            gen = [ [ 1.0 for i in range(0 , all_data.dim) ]\
                    for j in range(0 , d_per_dim) ]
            d_list.append(_np.array(gen))
        else:
            d_list.append([])
    K.update_delta(d_list)
    K.numbers()

    # if user has provided values, overwrite the above code
    if par.delta != []:
        K.update_delta(par.delta)
    if par.sigma != []:
        K.update_sigma(par.sigma)
    K.numbers()

    return None

### kernel bases class
class _kernel():
    def __init__(self, sigma, delta, nugget, name, delta_names, sigma_names, v=False, mv=False, cv=False):
        if v == False:  ## if not combining kernels
            self.var_od_list = [self.var_od,]
            self.var_md_list = [self.var_md,]
            self.covar_list = [self.covar,]
            self.nugget = [self.nugget,]
            self.numbers()
        else:  ## if combining kernels
            self.var_od_list = v
            self.var_md_list = mv
            self.covar_list = cv
            self.sigma = sigma
            self.delta = delta
            self.name = name
            self.delta_names = delta_names
            self.sigma_names = sigma_names
            self.numbers()
            self.nugget = nugget
        self.f = self.run_var_list

    def __add__(self, other):
        v = self.var_od_list + other.var_od_list
        mv = self.var_md_list + other.var_md_list
        cv = self.covar_list + other.covar_list
        sigma = self.sigma + other.sigma
        delta = self.delta + other.delta
        name = self.name + other.name
        delta_names = self.delta_names + other.delta_names
        sigma_names = self.sigma_names + other.sigma_names
        nugget = self.nugget + other.nugget
        return _kernel(sigma, delta, nugget, name, delta_names, sigma_names, v, mv, cv)

    ## calculates the covariance matrix (X,X) for each kernel and adds them 
    def run_var_list(self, X, no_noise=False):
        ## 1: calculates the off diagonal elements only
        ## 2: sums the off-diagonals (more efficient)
        ## 3: adds the missing main-diagonal values afterwards
        ##    (care must be taken to add the correct diagonal values)

        ## 1. add up the lower triangulars (will be missing the main diagonal)
        A = self.var_od_list[0](X,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1,len(self.var_od_list)):

            sub_A = self.var_od_list[c](X,\
                      self.sigma[c], self.delta[c], self.nugget[c])

            #if sub_A.size != 0 :
            if self.name[c] != "noise" and self.name[c] != "noisefit" :
                ## 2. sums the off-diagonals
                A = A + sub_A

        ## convert resulting matrix to squareform
        A = _dist.squareform(A)

        ## 3. now add the missing main diagonals
        diags = self.var_md_list[0](X,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1, len(self.var_md_list)) :

            sub_diag = self.var_md_list[c](X,\
                         self.sigma[c],self.delta[c],self.nugget[c])

            # when we build the matrix K(X*,X*) we should NOT included experimental noise
            if self.name[c] != "noise" and self.name[c] != "noisefit" :  # not a noise kernel
                diags = diags + sub_diag
            else: # is a noise kernel
                if no_noise == False : # check if we're adding the noise
                    diags = diags + sub_diag

        _np.fill_diagonal(A , diags)

        return A
   
    ## calculates the covariance matrix (X',X) for each kernel and adds them 
    def run_covar_list(self, XT, XV):
        res = self.covar_list[0](XT,XV,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1, len(self.covar_list)) :

            sub_res = self.covar_list[c](XT, XV, \
                        self.sigma[c], self.delta[c], self.nugget[0])

            #if sub_res.size != 0 :
            if self.name[c] != "noise" and self.name[c] != "noisefit" :
                res = res + sub_res
        return res
 
    ## updates the sigma belonging to each kernel
    def update_sigma(self, s):
        for i in range(0,len(self.name)):
            self.sigma[i] = _np.array(s[i])

    ## updates the delta belonging to each kernel
    def update_delta(self, d):
        for i in range(0,len(self.name)):
            self.delta[i] = _np.array(d[i])

    ## calculates number of sigmas and deltas
    def numbers(self):
        self.sigma_num = 0
        self.delta_num = 0
        for c in range(0,len(self.name)):

            self.sigma_num = self.sigma_num + self.sigma[c][:].size

            if self.delta[c].size != 0:
                self.delta_num = self.delta_num + self.delta[c][:].size

    ## print the kernels and their hyperparameters
    def print_kernel(self):
        print("")
        # loop over kernels
        for k in range(0, len(self.name)):
            print(self.name[k])

            # loop over different delta within a kernel
            for dn in range(0, len(self.delta_names[k])):
                print(" %-*s :" % (15, self.delta_names[k][dn]), self.delta[k][dn])

            # loop over different sigma within a kernel
            for sn in range(0, len(self.sigma_names[k])):
                print(" %-*s : %1.6f" % (15, self.sigma_names[k][sn], self.sigma[k][sn]))

## Gaussian (squared exponential) kernel - triggers MUCM llh
class gaussian_mucm(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array([1.0]) ,]
        self.name = ["gaussian_mucm",]
        self.delta_names = [["lengthscale"],]
        self.sigma_names = [["variance"],]
        self.nugget = nugget
        self.desc = "s0^2 exp{ -(X-X')^2 / d0^2 }"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)
    
    ## calculates only the off diagonals
    def var_od(self, X, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A)
        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        ## since nugget was never subtracted from this, doesn't need re-adding
        return s[0]**2

    ## calculates squareform (i.e. entire matrix) 
    def covar(self, XT, XV, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A)
        return A


## Gaussian (squared exponential) kernel - triggers GP4ML llh
class gaussian(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array([1.0]) ,]
        self.name = ["gaussian",]
        self.delta_names = [ ["lengthscale"] ,]
        self.sigma_names = [ ["variance"],]
        self.nugget = nugget
        self.desc = "s0^2 exp{ -(X-X')^2 / 2 d0^2 }"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)
    
    ## calculates only the off diagonals
    def var_od(self, X, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A/2.0)
        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        ## nugget was never subtracted from md, doesn't need re-adding
        return s[0]**2

    ## calculates squareform (i.e. entire matrix) 
    def covar(self, XT, XV, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A/2.0)
        return A


## pointwise noise
class noise(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array([]) ]
        self.name = ["noise",]
        self.delta_names = [[]]
        self.sigma_names = [ ["variance"],]
        self.nugget = nugget
        self.desc = "s0^2"
        print(self.name[0] , self.desc)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)
    ## no off diagonal terms for noise kernel
    def var_od(self, X, s, d, n):
        return 0
    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        return s[0]**2
    def covar(self, XT, XV, s, d, n):
        return 0


## linear kernel s0^2 * (X - d)(X' - d) + s1^2
class linear(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [[1.0],[1.0]] ) ,] ## 2 sigma
        self.delta = [ _np.array([1.0]) ,]
        self.name = ["linear",]
        self.delta_names = [["offset"] ,]
        self.sigma_names = [ ["var_signal","var_noise"],]
        self.nugget = nugget
        self.desc = "s0^2 * (X - d0)(X' - d0) + s1^2"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)

    # idx into the condensed array
    #def idx(self, i, j, N):
    #    return int( i*N + j - i*(i+1)/2 - i - 1 )

    def var_od(self, X, s, d, n):
        N = X[:,0].size
        A = _np.empty([int(N * (N-1) / 2)]) 
        s2 = s[0]**2

        X = X - d

        # fill only the upper triangle of the array
        k = 0
        for i in range(0, N-1):
            for j in range(i+1, N):
                #A[self.idx(i,j,N)] = X[i]*X[j]
                A[k] = X[i].dot(X[j])
                k = k + 1

        A = (s2*(1.0-n))*A
        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        N = X[:,0].size
        A = _np.empty([N])
        s2 = s[0]**2
        sn2 = s[1]**2

        X = X - d

        # only add noise on the diagonal, right?
        for k in range(0, N):
            A[k] =  s2*(X[k].dot(X[k])) + sn2
        return A

    def covar(self, XT, XV, s, d, n):
        NT = XT[:,0].size
        NV = XV[:,0].size
        A = _np.empty([NT, NV])
        s2 = s[0]**2

        XT = XT - d
        XV = XV - d

        for i in range(0, NT):
            for j in range(0, NV):
                A[i,j] = XT[i].dot( XV[j] )

        A = (s2*(1.0-n))*A
        return A


## rational quadratic (s_0)^2 * (1 + (X-X')^2 / 2 (s_1)^2 (s_2))^(-s_2)
## this rational quadratic has a single lengthscale s_1
class rat_quad(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [[1.0],[1.0],[1.0]] ) ,] ## 3 sigma
        self.delta = [ _np.array([]) ]
        self.name = ["rational_quadratic",]
        self.delta_names = [[], ]
        self.sigma_names = [ ["variance" , "lengthscale", "alpha"],]
        self.nugget = nugget
        self.desc = "s0^2 { 1 + (X-X')^2 / 2 s1^2 s2 }^(-s2)"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)

    def var_od(self, X, s, d, n):
        N = X[:,0].size
        A = _np.empty([int(N * (N-1) / 2)]) 
        s0_2 = s[0]**2
        s1_2 = s[1]**2

        # fill only the upper triangle of the array
        k = 0
        for i in range(0, N-1):
            for j in range(i+1, N):
                dx = X[i] - X[j]
                A[k] =  1.0 / ( (1.0 \
                          + ( dx ).dot( dx ) \
                              / (2.0 * s1_2 * s[2]) \
                                    )**(s[2]) )
                k = k + 1

        A = (s0_2*(1.0-n))*A
        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        N = X[:,0].size
        A = _np.empty([N])

        for k in range(0, N):
            A[k] =  (s[0]**2) / (1.0**s[2])
        return A

    def covar(self, XT, XV, s, d, n):
        NT = XT[:,0].size
        NV = XV[:,0].size
        A = _np.empty([NT, NV])
        s0_2 = s[0]**2
        s1_2 = s[1]**2

        for i in range(0, NT):
            for j in range(0, NV):
                dx = XT[i] - XV[j]
                A[i,j] =  1.0 / ( (1.0 \
                          + ( dx ).dot( dx ) \
                              / (2.0 * s1_2 * s[2]) \
                                    )**s[2] )

        A = (s0_2*(1.0-n))*A
        return A


## this periodic has 1 lengthscale per dim and takes the sine of scaled distance|Pi*(X-X')/p|  ... this is opposed to having a sum of sines within the Gaussian, which would allow for 2 lengthscales per dim (as there could be a separate lengthscale to divide each sine by) 
class periodic(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [[1.0],[1.0]] ) ,]
        self.delta = [ _np.array( [1.0] ) ,]
        self.name = ["periodic",]
        self.delta_names = [["period"] ,]
        self.sigma_names = [ ["variance", "overall_lenthscale"],]
        self.nugget = nugget
        self.desc = "s0^2 exp{ - 2 sin^2 [ pi (X - X') / d0 ] / s1^2  }"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)

    def var_od(self, X, s, d, n):
        l = 1.0 / s[1]**2
        p = 1.0 / d[0]
        s2 = s[0]**2
        A = _dist.pdist((_np.pi*p)*X,'euclidean')
        #A = (s2)*_np.exp(-2.0*l*_np.sin(A)**2)
        #A = (s2)*_np.exp(-2.0*l*_np.sin(_np.pi*A*p)**2)

        A = (s2*(1.0-n))*_np.exp(-2.0*l*_np.sin(A)**2)

        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        return s[0]**2

    def covar(self, XT, XV, s, d, n):
        l = 1.0 / s[1]**2
        p = 1.0 / d[0]
        s2 = s[0]**2
        A = _dist.cdist((_np.pi*p)*XT, (_np.pi*p)*XV, 'euclidean')
        #A = (s2)*_np.exp(-2.0*l*_np.sin(A)**2)
        #A = (s2)*_np.exp(-2.0*l*_np.sin(_np.pi*A*p)**2)

        A = (s2*(1.0-n))*_np.exp(-2.0*l*_np.sin(A)**2)
        return A


## this periodic X gaussian has 2 lengthscale per dim and takes the sine of scaled distance|Pi*(X-X')/p|  ...
class periodic_decay(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [[1.0],[1.0]] ) ,]
        self.delta = [ _np.array( [[1.0],[1.0]] ) ,]
        self.name = ["periodic_decay",]
        self.delta_names = [ ["period", "decay_length"] ,]
        self.sigma_names = [ ["variance", "overall_lenthscale"],]
        self.nugget = nugget
        self.desc = "s0^2 exp{-2sin^2[pi(X-X')/d0]/s1^2 -(X-X')^2/d1^2}"
        self.nug_str = "(v = "+str(self.nugget)+")" if self.nugget!=0 else ""
        print(self.name[0] , self.desc , self.nug_str)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)

    def var_od(self, X, s, d, n):
        # Periodic Hyperparamater
        l = 1.0 / s[1]**2
        p = 1.0 / d[0]
        # Gaussian hyperparameter
        w = 1.0 / d[1]**2
        s2 = s[0]**2

	# calculate the periodic part
        P = _dist.pdist((_np.pi*p)*X,'euclidean')

	# calculate the gaussian part
        G = _dist.pdist(X*w,'sqeuclidean')

        A = (s2*(1.0-n))*_np.exp( -2.0*l*_np.sin(P)**2 - G/2.0 )

        return A

    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        return s[0]**2

    def covar(self, XT, XV, s, d, n):
        # Periodic Hyperparamater
        l = 1.0 / s[1]**2
        p = 1.0 / d[0]
        # Gaussian hyperparameter
        w = 1.0 / d[1]**2
        s2 = s[0]**2

	# calculate the periodic part
        P = _dist.cdist((_np.pi*p)*XT, (_np.pi*p)*XV, 'euclidean')

	# calculate the gaussian part
        G = _dist.cdist(XT*w, XV*w, 'sqeuclidean')

        A = (s2*(1.0-n))*_np.exp( -2.0*l*_np.sin(P)**2 - G/2.0 )

        return A


## this example kernel demonstrates 2 delta per input dimension
## and two sigma
## it simply uses the second delta per dim for a Gaussian kernel
## the second sigma is pointwise noise
class two_delta_per_dim(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [[1.0],[1.0]] ) ,] ## 2 sigma
        self.delta = [ _np.array( [[1.0],[1.0]] ) ,] ## 2 delta per dim
        self.name = ["two_delta_per_dim",]
        self.delta_names = [ ["lengthscale", "lengthscale"],]
        self.sigma_names = [ ["var_sig", "var_noise"],]
        self.nugget = nugget
        print(self.name ,"( Nug", self.nugget,")")
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)

    def var_od(self, X, s, d, n):
        ## for this example we'll only use first delta and use Gaussian kernel
        w = 1.0/d[1]
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A)
        return A

    def var_md(self, X, s, d, n):
        return s[0]**2 + s[1]**2

    def covar(self, XT, XV, s, d, n):
        w = 1.0/d[1]
        s2 = s[0]**2
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (s2*(1.0-n))*_np.exp(-A)
        return A


########################
# user defined kernels #
########################]

## noise made from basis functions
class noisefit(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array( [1.0] ) ,]
        self.delta = [ _np.array( [[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]] ) ]
        self.name = ["noisefit",]
        self.delta_names = [["beta0", "beta1", "beta2", "beta3", "beta4", "beta5"]]
        self.sigma_names = [ ["variance"],]
        self.nugget = nugget
        self.desc = "s0^2 exp(sum(d*phi(x)))"
        print(self.name[0] , self.desc)
        _kernel.__init__(self, self.sigma, self.delta, self.delta_names, self.sigma_names, self.nugget, self.name)
    ## no off diagonal terms for noise kernel
    def var_od(self, X, s, d, n):
        return 0
    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        # basis function definition
        (b0, b1, b2, b3, b4, b5) = d
        # chebyshev polynomials
        T0 = lambda x: 1.0
        T1 = lambda x: x
        T2 = lambda x: 2.0*x**2 - 1
        T3 = lambda x: 4.0*x**3 - 3.0*x
        T4 = lambda x: 8.0*x**4 - 8.0*x**2 + 1
        T5 = lambda x: 16.0*x**5 - 20.0*x**3 + 5*x
        # chebyshev polynomials
 #       T0 = lambda x: 1.0
 #       T1 = lambda x: x
 #       T2 = lambda x: x**2
 #       T3 = lambda x: x**3
 #       T4 = lambda x: x**4
 #       T5 = lambda x: x**5
        # function
        phi = lambda x: b0*T0(x) + b1*T1(x) + b2*T2(x) + b3*T3(x) + b4*T4(x) + b5*T5(x)
        ## returning an array should still use the diagonal function properly
        return s[0]**2 * _np.exp( phi(X) )
    def covar(self, XT, XV, s, d, n):
        return 0
