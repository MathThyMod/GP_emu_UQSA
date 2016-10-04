##############################
#### emulatorfunctions.py ####
##############################

import numpy as _np
import gp_emu._emulatorclasses as __emuc
import gp_emu._emulatoroptimise as __emuo
import gp_emu.emulatorkernels as __emuk
import gp_emu.emulatorplotting as __emup


### builds the entire emulator and training structures
def setup(config_file, datashuffle=True, scaleinputs=True):
    """Do initialisation of classes Beliefs, Hyperparams, Basis, TV_config, All_Data, Data, Posterior, Optimize, and K. Return instance of Emulator class.

    Args:
        config_file (str): Name of configuration file.
        datashuffle (bool): Default is True. Randomly orders dataset.
        scaleinputs (bool): Default is True. Scales inputs into range 0 to 1.

    Returns:
        Emulator: Initialised Emulator class.

    """

    # returns instance of configuration
    config = __emuc.Config(config_file)

    #### read from beliefs file
    beliefs = __emuc.Beliefs(config.beliefs)
    par = __emuc.Hyperparams(beliefs)
    basis = __emuc.Basis(beliefs)

    #### split data T & V ; (k,c,noV) - no.sets, set for V, no.V.sets
    tv_conf = __emuc.TV_config(*(config.tv_config))
    #tv_conf = __emuc.TV_config(*(config.tv_config+["False"]))
    all_data = __emuc.All_Data(config.inputs,config.outputs,tv_conf,beliefs,par, datashuffle, scaleinputs)

    ## build the kernel based on beliefs file
    K = __emuk.build_kernel(beliefs)

    __emuk.auto_configure_kernel(K, par, all_data)

    #### build all __emuclator structures from beliefs and data
    ## FIX THIS FOR PYTHON 2.7
    (x_T, y_T) = all_data.choose_T()
    (x_V, y_V) = all_data.choose_V()
    training = __emuc.Data(x_T, y_T, basis, par, beliefs, K)
    validation = __emuc.Data(x_V, y_V, basis, par, beliefs, K)
    post = __emuc.Posterior(validation, training, par, beliefs, K)
    opt_T = __emuo.Optimize(training, basis, par, beliefs, config)
    
    return __emuc.Emulator\
        (config, beliefs,par,basis,tv_conf,all_data,training,validation,post,opt_T, K)


### trains and validates while there is still validation data left
def train(E, auto=True, message=False):
    """Do training of emulator hyperparameters on the training dataset and validate against the first validation dataset. Additional rounds of training, in which each validation dataset is included in the training dataset, may be done.

    Args:
        E (Emulator): Emulator instance.
        auto (bool): Default is True. Automatically retrain with last validation set included in training set, and validate against next training set.
        message (bool): Default is False. Print message from fitting routines.

    Returns:
        None

    """

    config = E.config

    if auto:
        E.tv_conf.auto_train()
    else:
        E.tv_conf.auto = False

    while E.tv_conf.doing_training():
        E.opt_T.llhoptimize_full\
          (config.tries,config.constraints,config.bounds,config.stochastic,message)

        E.training.remake()
        E.validation.remake()
        E.post.remake()

        E.post.mahalanobis_distance()
        E.post.indiv_standard_error(ise=2.0)

        E.beliefs.final_beliefs(E, config, E.par, E.all_data.minmax, E.K, False)
        E.post.final_design_points(E, config, False)

        if E.tv_conf.check_still_training():
            E.post.incVinT()
            E.tv_conf.next_Vset()
            E.all_data.choose_new_V(E.validation)
            E.training.remake()
            E.validation.remake()
            E.post.remake()


    if E.tv_conf.do_final_build():
        print("\n***Doing final build***")

        E.post.incVinT()
        E.training.remake()
        E.opt_T.llhoptimize_full\
          (config.tries,config.constraints,config.bounds,config.stochastic, message)
        E.training.remake()

        E.beliefs.final_beliefs(E, config, E.par, E.all_data.minmax, E.K, True)
        E.post.final_design_points(E, config, True)

    return None


### plotting function 
def plot(E,plot_dims,fixed_dims,fixed_vals,mean_or_var="mean",customLabels=[]):
    """Do plot of the Emulator posterior against 1 or 2 input variables, while holding the other inputs at constant values.

    Args:
        E (Emulator): Emulator instance.
        plot_dims (int list): Dimensions of inputs to plot (1 or 2 list items).
        fixed_dims (int list): Dimensions of inputs to hold fixed.
        fixed_vals (float list): Values of the inputs that aren't being plotted.
        mean_or_var (string): Choose to plot mean ("mean") of variance ("var").
        customLabels (string list): Labels ["x","y"] for the x and y axes.

    Returns:
        None

    """
    dim = E.training.inputs[0].size
    #if input("\nPlot full prediction? y/[n]: ") == 'y':
    print("***Generating full prediction***")
    if len(plot_dims) == 1 and dim>1:
        one_d = True
        if customLabels == []:
            xlabel="input " + str(plot_dims[0])
            ylabel="output " + str(E.beliefs.output)
        else:
            xlabel=customLabels[0]
            ylabel=customLabels[1]
    else:
        one_d =False
        if customLabels == []:
            xlabel="input " + str(plot_dims[0])
            if dim == 1:
                ylabel="output "
            else:
                ylabel="input " + str(plot_dims[1])
        else:
            xlabel=customLabels[0]
            ylabel=customLabels[1]
    pn=30 ### large range of x i.e. pnXpn points
    # which dims to 2D plot, list of fixed dims, and values of fixed dims
    __emup.full_xrange = __full_input_range(dim, pn, pn,\
        plot_dims, fixed_dims, fixed_vals, one_d)
    predict = __emuc.Data(full_xrange, 0, E.basis, E.par, E.beliefs, E.K) # don't pass y
    post = __emuc.Posterior(predict, E.training, E.par, E.beliefs, E.K) # calc post with x as V
    __emup.plotting(dim, post, pn, pn, one_d, mean_or_var, labels=[xlabel,ylabel]) ## plot

    return None
