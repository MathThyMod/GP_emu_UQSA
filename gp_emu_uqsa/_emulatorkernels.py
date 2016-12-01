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
        N = X[:,0].size
        f = _np.empty([int(N * (N-1) / 2)])

        # fill only the upper triangle of the array
        k = 0
        for i in range(0, N-1):
            for j in range(i+1, N):
                f[k] = ((X[i,di] - X[j,di])/self.d[di])**2 
                k = k + 1
        
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

