from __future__ import print_function
import numpy as _np
import matplotlib.pyplot as _plt


# generate a range of inputs for use in prediction using posterior
def make_inputs(dim, rows, cols, plot_dims, fixed_dims, fixed_vals, one_d):
    if dim>=2:
        # if doing a 2D contour plot
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
        # if doing a 1D line plot
        else: 
            RF = rows*cols
            X1 = _np.linspace(0.0,1.0,RF)
            x_all=_np.zeros((RF,dim))
            x_all[:,plot_dims[0]] = X1
            if dim>1:
                for i in range(0,len(fixed_dims)):
                    x_all[:,fixed_dims[i]] = fixed_vals[i]
    # if 1D inputs
    else:
        RF = rows*cols
        X1 = _np.linspace(0.0,1.0,RF)
        x_all=_np.zeros((RF,1))
        x_all[:,0] = X1

    return x_all


# function plotting function
def plotting(dim, post, rows, cols, one_d, mean_or_var, labels=[]):
    # decide what to plot
    if mean_or_var != "var":
        prediction=post.newnewmean
    else:
        prediction=_np.diag(post.newnewvar)

    if dim>=2 and one_d!=True:
        RF = rows
        CF = cols

        # set up Z-axis (post) and lower and upper intervals
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
      
        # set the labels 
        _plt.xlabel(labels[0])
        _plt.ylabel(labels[1])
 
        im = _plt.imshow(ZF.T, origin = 'lower',\
             cmap = _plt.get_cmap('rainbow'), extent = (0.0,1.0,0.0,1.0))
        _plt.colorbar()
        _plt.show()
    else:
        RF = rows*cols

        # set up Z-axis (post) and lower and upper intervals
        ZF = _np.zeros((RF))
        LF = _np.zeros((RF))
        UF = _np.zeros((RF))
        for i in range(0,RF):
            ZF[i]=prediction[i]
            LF[i]=post.LI[i]
            UF[i]=post.UI[i]

        print("Plotting... output range:", _np.amin(ZF), "to" , _np.amax(ZF))
        _plt.plot(_np.linspace(0.0,1.0,RF), ZF, linewidth=2.0)

        # set the labels 
        _plt.xlabel(labels[0])
        _plt.ylabel(labels[1])

        _plt.show()

    return None


