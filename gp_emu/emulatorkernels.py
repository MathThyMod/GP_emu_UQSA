import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class _Kernel():
    def __init__(self, sigma, delta, nugget, name, v=False, cv=False):
        if v == False:  ## if not combining kernels
            self.var_list = [self.var,]
            self.covar_list = [self.covar,]
            self.nugget = [self.nugget,]
            self.numbers()
        else:  ## if combining kernels
            self.var_list = v
            self.covar_list = cv
            self.sigma = sigma
            self.delta = delta
            self.name = name
            self.numbers()
            self.nugget = nugget
        self.f = self.run_var_list

    def __add__(self, other):
        v = self.var_list + other.var_list
        cv = self.covar_list + other.covar_list
        sigma = self.sigma + other.sigma
        delta = self.delta + other.delta
        name = self.name + other.name
        nugget = self.nugget + other.nugget
        return _Kernel(sigma, delta, nugget, name, v, cv)

    def run_var_list(self, X):
        res = self.var_list[0](X,self.sigma[0],self.delta[0],self.nugget[0])
        for c in range(1,len(self.var_list)): ## each i uses a covar func
            res=res+self.var_list[c](X,self.sigma[c],self.delta[c],self.nugget[c])
        return res
    
    def run_covar_list(self, XT, XV):
        res = self.covar_list[0](XT,XV,self.sigma[0],self.delta[0])
        for c in range(1,len(self.covar_list)): ## each i uses a covar func
            if self.name[c] != "Noise": ## noise in covar returns 0 anyway
                res=res+self.covar_list[c](XT,XV,self.sigma[c],self.delta[c])
        return res
 
    def update_sigma(self, s):
        for i in range(0,len(self.name)):
            self.sigma[i] = np.array(s[i])

    def update_delta(self, d):
        for i in range(0,len(self.name)):
            self.delta[i] = np.array(d[i])

    def numbers(self):
        self.sigma_num = 0
        self.delta_num = 0
        for c in range(0,len(self.name)): ## each i uses a covar func
            self.sigma_num = self.sigma_num + self.sigma[c][:].size
            if self.name[c] != "Noise":
                self.delta_num = self.delta_num + self.delta[c][:].size

class Gaussian(_Kernel):
    def __init__(self, nugget=0):
        self.sigma = [ np.array([1.0]) ,]
        self.delta = [ np.array([1.0]) ,]
        self.name = ["Gaussian",]
        self.nugget=nugget
        print(self.name ,"( + Nugget:", self.nugget,")")
        _Kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    def var(self, X, s, d, n):
        w = 1.0/d
        A = pdist(X*w,'sqeuclidean')
        A = squareform(A)
        if n!=0:
            A = n*np.identity(X[:,0].size) + (1.0-n)*A
        A = (s[0]**2)*np.exp(-A)
        return A
    def covar(self, XT, XV, s, d):
        w = 1.0/d
        A = cdist(XT*w,XV*w,'sqeuclidean')
        ## cdist already gives squareform
        A = (s[0]**2)*np.exp(-A)
        return A

class Test(_Kernel):
    def __init__(self, nugget=0):
        self.sigma = [ np.array([1.0]) ,]
        self.delta = [ np.array( [[1.0],[1.0]] ) ,] ## 2 delta per dim
        self.name = ["Gaussian",]
        self.nugget=nugget
        print(self.name ,"( + Nugget:", self.nugget,")")
        _Kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    def var(self, X, s, d, n):
        w = 1.0/d[0]
        A = pdist(X*w,'sqeuclidean')
        A = squareform(A)
        if n!=0:
            A = n*np.identity(X[:,0].size) + (1.0-n)*A
        A = (s[0]**2)*np.exp(-A)
        return A
    def covar(self, XT, XV, s, d):
        w = 1.0/d[0]
        A = cdist(XT*w,XV*w,'sqeuclidean')
        ## cdist already gives squareform
        A = (s[0]**2)*np.exp(-A)
        return A

class Noise(_Kernel):
    def __init__(self, nugget=0):
        self.sigma = [ np.array([0.0]) ,]
        self.delta = [ np.array([]) ]
        self.name = ["Noise",]
        self.nugget=nugget
        print(self.name)
        _Kernel.__init__(self, self.sigma, self.delta, self.nugget, self.name)
    def var(self, X, s, d, n):
        A = (s[0]**2)*np.identity(X[:,0].size)
        return A
    def covar(self, XT, XV, s, d):
        A = np.zeros((XT[:,0].size,XV[:,0].size))
        return A 

