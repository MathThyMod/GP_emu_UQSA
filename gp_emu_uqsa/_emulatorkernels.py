from __future__ import print_function
import numpy as _np
import scipy.spatial.distance as _dist

_np.set_printoptions(precision=6)
_np.set_printoptions(suppress=True)


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
        self.A = (1.0-self.n)*_np.exp(-self.A)
        self.A = _dist.squareform(self.A)
        if predict: # 'predict' adds nugget back onto diagonal
            _np.fill_diagonal(self.A , 1.0)
        else: # 'estimate' - does not add nugget back onto diagonal
            _np.fill_diagonal(self.A , 1.0 - self.n)
        return self.A

    #def grad_delta_A(self, X, di):
    #    N = X[:,0].size
    #    f = _np.empty([int(N * (N-1) / 2)])

    #    w = 1.0/(2.0*self.d[di]**3)

    #    # fill only the upper triangle of the array
    #    k = 0
    #    for i in range(0, N-1):
    #        for j in range(i+1, N):
    #            f[k] = X[i,di] * X[j,di] * w 
    #            k = k + 1

    #    f = _dist.squareform(f)

    #    f = self.A * f
    #    return f


    #def grad_nugget_A(self, X):
    #    #w = 1.0/self.d
    #    #s2 = self.s**2
    #    #f = _dist.pdist(X*w,'sqeuclidean')
    #    #f = (s2*(-1.0))*_np.exp(-f/2.0)
    #    #f = _dist.squareform(f)
    #    # just zeros on diagonal now...
    #    
    #    ## pretty sure I could reverse engineer from A instead
    #    f = _np.copy(self.A)
    #    _np.fill_diagonal(f, 0.0)
    #    f = f/(1.0-self.n)
    #    return f

        
    ## calculates the covariance matrix (X',X) 
    def covar(self, XT, XV):
        w = 1.0/self.d
        A = _dist.cdist(XT*w,XV*w,'sqeuclidean')
        A = (1.0-self.n)*_np.exp(-A)
        return A

