##############################
#### emulatorfunctions.py ####
##############################

import numpy as _np
import gp_emu_uqsa._emulatorclasses as __emuc
import gp_emu_uqsa._emulatoroptimise as __emuo
import gp_emu_uqsa._emulatorkernels as __emuk
import gp_emu_uqsa._emulatorplotting as __emup


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

    # read from beliefs file
    beliefs = __emuc.Beliefs(config.beliefs)
    par = __emuc.Hyperparams(beliefs)
    basis = __emuc.Basis(beliefs)

    # split data T & V ; (k,c,noV) - no.sets, set for first V, no.V.sets
    tv_conf = __emuc.TV_config(*(config.tv_config))
    all_data = __emuc.All_Data(\
      config.inputs, config.outputs, tv_conf,\
      beliefs, par, datashuffle, scaleinputs)

    # build the kernel
    K = __emuk.kernel(all_data.x_full[0].size, par)

    # build remaining structures
    (x_T, y_T) = all_data.choose_T()
    (x_V, y_V) = all_data.choose_V()
    training = __emuc.Data(x_T, y_T, basis, par, beliefs, K)
    validation = __emuc.Data(x_V, y_V, basis, par, beliefs, K)
    post = __emuc.Posterior(validation, training, par, beliefs, K)
    opt_T = __emuo.Optimize(training, basis, par, beliefs, config)
    
    return __emuc.Emulator(\
      config, beliefs, par, basis, tv_conf,\
      all_data, training, validation, post, opt_T, K)


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

    # set training to be automatic or not
    E.tv_conf.auto_train(auto)

    # while there are validation sets remaining
    while E.tv_conf.doing_training():
        print("\n*** Training round",E.tv_conf.no_of_trains,"***")
        print("Training points:", E.training.inputs[:,0].size)
        # optimise the hyperparameters
        E.opt_T.llh_optimize(message)

        # remake the data structures with best hyperparameters
        E.training.remake()
        E.validation.remake()
        E.post.remake()

        # perform validation diagnostics
        E.post.mahalanobis_distance()
        E.post.indiv_standard_error(ise=2.0)

        # save the emulator to new belief files and data files
        E.beliefs.final_beliefs(E, False)
        E.post.final_design_points(E, False)

        # prepare for next round of training if more validation data left
        if E.tv_conf.check_still_training():
            print("Preparing for next round of training...")
            E.post.incVinT()
            E.tv_conf.next_Vset()
            E.all_data.choose_new_V(E.validation)
            E.training.remake()
            E.validation.remake()
            E.post.remake()

    # do training without doing more validation
    if E.tv_conf.do_final_build():
        print("\n*** Doing final build ***")

        # only include validation in training if we had any validation sets
        if E.tv_conf.noV != 0:
            E.post.incVinT()
        E.training.remake()

        # optimise the hyperparameters and rebuild training data structure
        E.opt_T.llh_optimize(message)
        E.training.remake()

        # save the emulator to new belief files and data files
        E.beliefs.final_beliefs(E, True)
        E.post.final_design_points(E, True)

    return None


# plotting function 
def plot(E,
        plot_dims, fixed_dims, fixed_vals, mean_or_var="mean", customLabels=[], points=False):
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
    minmax = []
    # for plotting training points on a scatter graph
    x = []
    y = []
    print("\n*** Generating plot ***")

    # if we are doing a 1D plot for multidimensional inputs
    if len(plot_dims) == 1 and dim>1:
        one_d = True

        minmax.append( [_np.amin(E.training.inputs[:,plot_dims[0]]) , _np.amax(E.training.inputs[:,plot_dims[0]])] )

        if points and mean_or_var == "mean":
            x = E.training.inputs[:,plot_dims[0]]
            y = E.training.outputs

        # set labels
        if customLabels == []:
            xlabel="input " + str(plot_dims[0])
            ylabel="output " + str(E.beliefs.output)
        else:
            try:
                xlabel=customLabels[0]
            except IndexError as e:
                xlabel="input " + str(plot_dims[0])
            try:
                ylabel=customLabels[1]
            except IndexError as e:
                ylabel="output " + str(E.beliefs.output)

    else:
        one_d =False

        if points and mean_or_var == "mean":
            x = E.training.inputs[:,plot_dims[0]]
            y = E.training.outputs

        minmax.append( [_np.amin(E.training.inputs[:,plot_dims[0]]) , _np.amax(E.training.inputs[:,plot_dims[0]])] )
        if dim > 1:
            minmax.append( [_np.amin(E.training.inputs[:,plot_dims[1]]) , _np.amax(E.training.inputs[:,plot_dims[1]])] )
        # set labels
        if customLabels == []:
            xlabel="input " + str(plot_dims[0])
            if dim == 1:
                ylabel="output "
            else:
                ylabel="input " + str(plot_dims[1])
        else:
            try:
                xlabel=customLabels[0]
            except IndexError as e:
                xlabel="input " + str(plot_dims[0])
            try:
                ylabel=customLabels[1]
            except IndexError as e:
                if dim == 1:
                    ylabel="output "
                else:
                    ylabel="input " + str(plot_dims[1])


    # number of inputs along each prediction dim
    pn=30
    # generate range of inputs to make predictions
    full_xrange = __emup.make_inputs(dim, pn, pn,\
        plot_dims, fixed_dims, fixed_vals, one_d, minmax)
    predict = __emuc.Data(full_xrange, None, E.basis, E.par, E.beliefs, E.K)
    post = __emuc.Posterior(predict, E.training, E.par, E.beliefs, E.K, False)

    # call the actual plotting routine
    __emup.plotting(dim, post, pn, pn, one_d, mean_or_var, minmax , x, y, labels=[xlabel,ylabel])

    return None
