import numpy as _np
import gp_emu._emulatorclasses as __emuc
import matplotlib.pyplot as _plt

#########################################
### private functions for this module ###
#########################################

### configure kernel with enough delta for all the kernels 
def __auto_configure_kernel(K, par, all_data):
    dim = all_data.x_full[0].size
    d_list = []
    for d in range(0, len(K.var_list)):
        if K.name[d] != "noise":
            d_per_dim = int(K.delta[d].flat[:].size/K.delta[d][0].size)
            gen = [[1.0 for i in range(0,dim)] for j in range(0,d_per_dim)]
            d_list.append(_np.array(gen))
        else:
            d_list.append([])
    K.update_delta(d_list)
    K.numbers()

    ### if user has provided delta and sigma values, overwrite the above
    if par.delta != []:
        K.update_delta(par.delta)
    if par.sigma != []:
        K.update_sigma(par.sigma)


### rebuilds training, validation, and post
def __rebuild(t, v, p):
    ###### rebuild data structures ######
    print("Building data structures")
    t.remake()
    v.remake()
    p.remake()


#### save emulator information to files
def __new_belief_filenames(E, config, final=False):
    new_beliefs_file=\
      config.beliefs+"-"+str(E.tv_conf.no_of_trains)
    new_inputs_file=\
      config.inputs+"-"+str(E.tv_conf.no_of_trains)
    new_outputs_file=\
      config.outputs+"-"+str(E.tv_conf.no_of_trains)
    if final:
        new_beliefs_file=new_beliefs_file+"f"
        new_inputs_file=new_inputs_file+"f"
        new_outputs_file=new_outputs_file+"f"
    return(new_beliefs_file, new_inputs_file, new_outputs_file)


### full range of inputs to get full posterior -- call via plot
def __full_input_range(dim,rows,cols,plot_dims,fixed_dims,fixed_vals,one_d):
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
            x_all=_np.zeros((dim,RF))
            x_all[:,plot_dims[0]] = X1
            if dim>1:
                for i in range(0,len(fixed_dims)):
                    x_all[:,fixed_dims[i]] = fixed_vals[i]
    else:
        RF = rows*cols
        X1 = _np.linspace(0.0,1.0,RF)
        x_all=_np.zeros((1,RF))
        x_all[:,0] = X1
    return x_all


### plotting function - should not be called directly, call plot instead
def __plotting(dim, post, rows, cols, one_d, mean_or_var):
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
       
        _plt.plot(X1,ZF, linewidth=2.0)
        _plt.show()


###################################
#### user accessible functions ####
###################################


### returns the initialised config class
def config_file(f):
    print("config file:" , f)
    return __emuc.Config(f)


### builds the entire emulator and training structures
def setup(config, K):
    #### read from beliefs file
    beliefs = __emuc.Beliefs(config.beliefs)
    par = __emuc.Hyperparams(beliefs)
    basis = __emuc.Basis(beliefs)

    #### split data T & V ; (k,c,noV) - no.sets, set for V, no.V.sets
    tv_conf = __emuc.TV_config(*(config.tv_config))
    #tv_conf = __emuc.TV_config(*(config.tv_config+["False"]))
    all_data = __emuc.All_Data(config.inputs,config.outputs,tv_conf,beliefs,par)

    __auto_configure_kernel(K, par, all_data)

    #### build all __emuclator structures from beliefs and data
    ## FIX THIS FOR PYTHON 2.7
    (x_T, y_T) = all_data.choose_T()
    (x_V, y_V) = all_data.choose_V()
    training = __emuc.Data(x_T, y_T, basis, par, beliefs, K)
    validation = __emuc.Data(x_V, y_V, basis, par, beliefs, K)
    post = __emuc.Posterior(validation, training, par, beliefs, K)
    opt_T = __emuc.Optimize(training, basis, par, beliefs, config)
    
    return __emuc.Emulator\
        (beliefs,par,basis,tv_conf,all_data,training,validation,post,opt_T, K)


### trains and validates while there is still validation data left
def training_loop(E, config, auto=True):
    if auto:
        E.tv_conf.auto_train()
    else:
        E.tv_conf.auto = False

    while E.tv_conf.doing_training():
        E.opt_T.llhoptimize_full\
          (config.tries,config.constraints,config.bounds,config.stochastic)

        __rebuild(E.training, E.validation, E.post)

        E.post.mahalanobis_distance()
        E.post.indiv_standard_error(ise=2.0)

        (nbf,nif,nof) = __new_belief_filenames(E, config)
        E.beliefs.final_beliefs(nbf, E.par, E.all_data.minmax, E.K)
        E.post.final_design_points(nif,nof,E.all_data.minmax)

        if E.tv_conf.check_still_training():
            E.post.incVinT()
            E.tv_conf.next_Vset()
            E.all_data.choose_new_V(E.validation)
            __rebuild(E.training, E.validation, E.post)


### does final training (including validation data) and saves to files
def final_build(E, config, auto=True):
    if auto:
        E.tv_conf.auto_train()
    else:
        E.tv_conf.auto = False

    if E.tv_conf.do_final_build():
        print("\n***Doing final build***")

        E.post.incVinT()
        E.training.remake()
        E.opt_T.llhoptimize_full\
          (config.tries,config.constraints,config.bounds,config.stochastic)
        E.training.remake()

        (nbf,nif,nof) = __new_belief_filenames(E, config, True)
        E.beliefs.final_beliefs(nbf, E.par, E.all_data.minmax, E.K)
        E.post.final_design_points(nif,nof,E.all_data.minmax)


### plotting function 
def plot(E, plot_dims, fixed_dims, fixed_vals, mean_or_var="mean"):
    dim = E.training.inputs[0].size
    if input("\nPlot full prediction? y/[n]: ") == 'y':
        print("***Generating full prediction***")
        if len(plot_dims) == 1 and dim>1:
            one_d = True
        else:
            one_d =False
        pn=30 ### large range of x i.e. pnXpn points
        # which dims to 2D plot, list of fixed dims, and values of fixed dims
        full_xrange = __full_input_range(dim, pn, pn,\
            plot_dims, fixed_dims, fixed_vals, one_d)
        predict = __emuc.Data(full_xrange, 0, E.basis, E.par, E.beliefs, E.K) # don't pass y
        post = __emuc.Posterior(predict, E.training, E.par, E.beliefs, E.K) # calc post with x as V
        __plotting(dim, post, pn, pn, one_d, mean_or_var) ## plot
