from __future__ import print_function
import numpy as _np
import scipy.spatial.distance as _dist

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

    # construct list of required delta
    d_list = [ ]
    for d in range(0, len(K.name)):
        if K.delta[d].size != 0:
            d_per_dim = int(K.delta[d].flat[:].size / K.delta[d][0].size)
            gen = [ [ 1.0 for i in range(0 , all_data.dim) ]\
                    for j in range(0 , d_per_dim) ]
            d_list.append(_np.array(gen))
        else:
            d_list.append([])

    # update the values of delta in the kernel
    K.update_delta(d_list)
    K.numbers()

    # if user has provided value, overwrite the above code
    if par.delta != []:
        K.update_delta(par.delta)
    if par.sigma != []:
        K.update_sigma(par.sigma)
    K.numbers()

    return None

### kernel bases class
class _kernel():
    def __init__(self, sigma, delta, nugget, name, v=False, mv=False, cv=False):
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
        nugget = self.nugget + other.nugget
        return _kernel(sigma, delta, nugget, name, v, mv, cv)

    ## calculates the covariance matrix (X,X) for each kernel and adds them 
    def run_var_list(self, X):
        ## 1: calculates the off diagonal elements only
        ## 2: sums the off-diagonals (more efficient)
        ## 3: adds the missing main-diagonal values afterwards
        ##    (care must be taken to add the correct diagonal values)

        ## add up the lower triangulars (will be missing the main diagonal)
        A = self.var_od_list[0](X,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1,len(self.var_od_list)):

            sub_A = self.var_od_list[c](X,\
                      self.sigma[c], self.delta[c], self.nugget[c])

            if sub_A != 0 :
                A = A + sub_A

        ## convert resulting matrix to squareform
        A = _dist.squareform(A)

        ## now add the missing main diagonals
        diags = self.var_md_list[0](X,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1, len(self.var_md_list)) :

            diags = diags+\
                 self.var_md_list[c](X,\
                   self.sigma[c],self.delta[c],self.nugget[c])

        _np.fill_diagonal(A , diags)

        return A
   
    ## calculates the covariance matrix (X',X) for each kernel and adds them 
    def run_covar_list(self, XT, XV):
        res = self.covar_list[0](XT,XV,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1, len(self.covar_list)) :

            sub_res = self.covar_list[c](XT, XV, \
                        self.sigma[c], self.delta[c], self.nugget[0])

            if sub_res != 0 :
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


## Gaussian (squared exponential) kernel - triggers MUCM llh
class gaussian_mucm(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array([1.0]) ,]
        self.name = ["gaussian_mucm",]
        self.nugget=nugget
        print("Kernel:" , self.name ,"( + Nugget:", self.nugget,")")
        _kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    
    ## calculates only the off diagonals
    def var_od(self, X, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*(1.0-n)*_np.exp(-A)
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
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*((1.0-n)*_np.exp(-A))
        return A


## Gaussian (squared exponential) kernel - triggers GP4ML llh
class gaussian(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array([1.0]) ,]
        self.name = ["gaussian",]
        self.nugget=nugget
        print(self.name ,"( + Nugget:", self.nugget,")")
        _kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    
    ## calculates only the off diagonals
    def var_od(self, X, s, d, n):
        w = 1.0/d
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*(1.0-n)*_np.exp(-A)
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
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*((1.0-n)*_np.exp(-A))
        return A


## pointwise noise
class noise(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([0.0]) ,]
        self.delta = [ _np.array([]) ]
        self.name = ["noise",]
        self.nugget=nugget
        print(self.name)
        _kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    ## no off diagonal terms for noise kernel
    def var_od(self, X, s, d, n):
        return 0
    ## calculates only the main diagonal
    def var_md(self, X, s, d, n):
        return s[0]**2
    def covar(self, XT, XV, s, d, n):
        return 0


## this example kernel demonstrates 2 delta per input dimension
## it simply uses the first delta per dim for a Gaussian kernel
class two_delta_per_dim(_kernel):
    def __init__(self, nugget=0):
        self.sigma = [ _np.array([1.0]) ,]
        self.delta = [ _np.array( [[1.0],[1.0]] ) ,] ## 2 delta per dim
        self.name = ["two_delta_per_dim",]
        self.nugget=nugget
        print(self.name ,"( + Nugget:", self.nugget,")")
        _kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)

    def var_od(self, X, s, d, n):
        ## for this example we'll only use first delta and use Gaussian kernel
        w = 1.0/d[0]
        s2 = s[0]**2
        A = _dist.pdist(X*w,'sqeuclidean')
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*(1.0-n)*_np.exp(-A)
        return A

    def var_md(self, X, s, d, n):
        return s[0]**2

    def covar(self, XT, XV, s, d, n):
        w = 1.0/d[0]
        s2 = s[0]**2
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        if n == 0:
            A = (s2)*_np.exp(-A)
        else:
            A = (s2)*((1.0-n)*_np.exp(-A))
        return A
