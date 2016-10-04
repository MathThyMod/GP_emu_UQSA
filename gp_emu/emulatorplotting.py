from __future__ import print_function
from builtins import input
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


### full range of inputs to get full posterior -- call via plot
def full_input_range(dim,rows,cols,plot_dims,fixed_dims,fixed_vals,one_d):
    if dim>=2:
        if one_d!=True:
            RF = rows
            CF = cols
            X1 = _np.linspace(0.0,1.0,RF)
            X2 = _np.linspace(0.0,1.0,CF)
            x_all=_np.zeros((RF*CF,dim))
            for i in range(0,RF):
                for j in range(0,CF):
                    x_all[i*CF+j,plot_dims[0]] = X1[i]
                    x_all[i*CF+j,plot_dims[1]] = X2[j]
            if dim>2:
                for i in range(0,len(fixed_dims)):
                    x_all[:,fixed_dims[i]] = fixed_vals[i]
        else:
            RF = rows*cols
            X1 = _np.linspace(0.0,1.0,RF)
            #x_all=_np.zeros((dim,RF))
            x_all=_np.zeros((RF,dim))
            #print(X1.shape)
            #print(x_all[:,plot_dims[0]].shape)
            x_all[:,plot_dims[0]] = X1
            if dim>1:
                for i in range(0,len(fixed_dims)):
                    x_all[:,fixed_dims[i]] = fixed_vals[i]
    else:
        RF = rows*cols
        X1 = _np.linspace(0.0,1.0,RF)
        #x_all=_np.zeros((1,RF))
        x_all=_np.zeros((RF,1))
        x_all[:,0] = X1

    return x_all


### plotting function - should not be called directly, call plot instead
def plotting(dim, post, rows, cols, one_d, mean_or_var, labels=[]):
    if dim>=2 and one_d!=True:
        RF = rows
        CF = cols
        ## these are the full predicions in a form that can be plotted
        X1 = _np.linspace(0.0,1.0,RF)
        X2 = _np.linspace(0.0,1.0,CF)
        x_all=_np.zeros((RF*CF,dim))
        for i in range(0,RF):
            for j in range(0,CF):
                x_all[i*CF+j,0] = X1[i]
                x_all[i*CF+j,1] = X2[j] 
        XF, YF = _np.meshgrid(X1, X2)
        if mean_or_var != "var":
            prediction=post.newnewmean
        else:
            prediction=_np.diag(post.newnewvar)
        ZF = _np.zeros((RF,CF))
        LF = _np.zeros((RF,CF))
        UF = _np.zeros((RF,CF))
        for i in range(0,RF):
            for j in range(0,CF):
                ZF[i,j]=prediction[i*CF+j]
                LF[i,j]=post.LI[i*CF+j]
                UF[i,j]=post.UI[i*CF+j]

        print("Plotting... output range:", _np.amin(ZF), "to" , _np.amax(ZF))
        fig = _plt.figure()
       
        _plt.xlabel(labels[0])
        _plt.ylabel(labels[1])
 
        im = _plt.imshow(ZF.T, origin='lower',\
             cmap=_plt.get_cmap('rainbow'), extent=(0.0,1.0,0.0,1.0))
        _plt.colorbar()
        _plt.show()
    else:
        RF = rows*cols
        ## these are the full predicions in a form that can be plotted
        X1 = _np.linspace(0.0,1.0,RF)
        if mean_or_var != "var":
            prediction=post.newnewmean
        else:
            prediction=_np.diag(post.newnewvar)
        ZF = _np.zeros((RF))
        LF = _np.zeros((RF))
        UF = _np.zeros((RF))
        for i in range(0,RF):
                ZF[i]=prediction[i]
                LF[i]=post.LI[i]
                UF[i]=post.UI[i]

        print("Plotting... output range:", _np.amin(ZF), "to" , _np.amax(ZF))
        #fig = _plt.figure()
       
        _plt.xlabel(labels[0])
        _plt.ylabel(labels[1])

        _plt.plot(X1,ZF, linewidth=2.0)
        _plt.show()

    return None


