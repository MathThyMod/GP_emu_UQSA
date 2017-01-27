from __future__ import print_function
import numpy as _np
import scipy.spatial.distance as _dist
import time

_np.set_printoptions(precision=6)
_np.set_printoptions(suppress=True)

## (1-nugget)*exp(~) + nugget(if on diagonal)
class kernel():
    def __init__(self, dim, par):
        self.d = par.delta
        self.n = par.nugget
   
    def set_hp(self, d, s, n):
        self.d = d
        self.n = n
        return

    def set_params(self, x):
        self.d = x[0:self.d.size]
        if x.size > self.d.size:
            self.n = x[-1]
        return

    def print_kernel(self):
        print("delta:", self.d)
        print("nugget:", self.n)

    ## tranforms the variables before sending to optimisation routines
    def transform(self, hp):
        return 2.0*_np.log(hp)

    ## untranforms the variables after sending to optimisation routines
    def untransform(self, hp):
        return _np.exp(hp/2.0)
 
    ## calculates the covariance matrix (X,X)
    def var(self, X, predict=True):
        w = 1.0/self.d
        self.A = _dist.pdist(X*w,'sqeuclidean')
        self.exp_save = _np.exp(-self.A)
        self.A = (1.0-self.n)*self.exp_save
        #self.A = (1.0-self.n)*_np.exp(-self.A)
        self.A = _dist.squareform(self.A)
        if predict: # 'predict' adds nugget back onto diagonal
            _np.fill_diagonal(self.A , 1.0)
        else: # 'estimate' - does not add nugget back onto diagonal
            _np.fill_diagonal(self.A , 1.0 - self.n)
        return self.A

    ## derivative wrt delta
    def grad_delta_A(self, X, di, s2):
        N = X.size
        w = 1.0 / self.d[di]

        f = _dist.pdist((X*w).reshape(N,1),'sqeuclidean')
 
        f = ((1.0-self.n)*s2)*f*self.exp_save
        f = _dist.squareform(f)
        ## because of prefactor, diagonal will be zeros

        return f


    def grad_nugget_A(self, X, s2):
        f = (0.5*(-self.n)*s2)*self.exp_save
        f = _dist.squareform(f)
        ## don't add 1.0 onto the diagonal here 

        return f

        
    ## calculates the covariance matrix (X',X) 
    def covar(self, XT, XV):
        w = 1.0/self.d
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (1.0-self.n)*_np.exp(-A)
        return A


## exp(~) + nugget**2(if on diagonal)
class kernel_alt_nug():
    def __init__(self, dim, par):
        self.d = par.delta
        self.n = par.nugget
   
    def set_hp(self, d, s, n):
        self.d = d
        self.n = n
        return

    def set_params(self, x):
        self.d = x[0:self.d.size]
        if x.size > self.d.size:
            self.n = x[-1]
        return

    def print_kernel(self):
        print("delta:", self.d)
        print("nugget:", self.n)

    ## tranforms the variables before sending to optimisation routines
    def transform(self, hp):
        return 2.0*_np.log(hp)

    ## untranforms the variables after sending to optimisation routines
    def untransform(self, hp):
        return _np.exp(hp/2.0)
 
    ## calculates the covariance matrix (X,X)
    def var(self, X, predict=True):
        w = 1.0/self.d
        self.A = _dist.pdist(X*w,'sqeuclidean')
        self.exp_save = _np.exp(-self.A)
        self.A = self.exp_save
        #self.A = (1.0-self.n)*_np.exp(-self.A)
        self.A = _dist.squareform(self.A)
        if predict: # 'predict' adds nugget onto diagonal
            _np.fill_diagonal(self.A , 1.0 + self.n**2)
        else: # 'estimate' - does not add nugget onto diagonal
            _np.fill_diagonal(self.A , 1.0)
        return self.A

    ## derivative wrt delta
    def grad_delta_A(self, X, di, s2):
        N = X.size
        w = 1.0 / self.d[di]

        f = _dist.pdist((X*w).reshape(N,1),'sqeuclidean')
 
        f = (s2)*f*self.exp_save
        f = _dist.squareform(f)
        ## because of prefactor, diagonal will be zeros

        return f


    def grad_nugget_A(self, X, s2):
        N = X[:,0].size
        f = _np.zeros((N,N))
        _np.fill_diagonal(f, (self.n**2)*s2)

        return f

        
    ## calculates the covariance matrix (X',X) 
    def covar(self, XT, XV):
        w = 1.0/self.d
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = _np.exp(-A)
        return A

