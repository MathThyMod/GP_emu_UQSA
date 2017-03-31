import gp_emu_uqsa as g
import numpy as np
import gp_emu_uqsa._emulatorclasses as __emuc
import gp_emu_uqsa.design_inputs as _gd


#### transforms the noise before fitting

### transform z = log(y)
def __transform(x, reg="log"):
    if reg == "log":
        return np.log(x)
    else:
        return x

### untransform y = exp(z)
def __untransform(x, reg="log"):
    if reg == "log":
        return np.exp(x)
    else:
        return x

def __read_file(ifile):
    print("*** Reading file:", ifile ,"***")
    dct = {}
    try:
        with open(ifile, 'r') as f:
            for line in f:
                (key, val) = line.split(' ',1)
                dct[key] = val.strip()
        return dct
    except OSError as e:
        print("ERROR: Problem reading file.")
        exit()


# currently works only for 1D data
def noisefit(data, noise, stopat=20, olhcmult=100, samples=200):
    """Try to fit one emualtor to the mean of the data and another emulator to the noise of the data. Results of estimating the noise are saved to the files 'noise-inputs' and 'noise-outputs'.

    Args:
        data (str): Name of configuration file for fitting the input-output data.
        noise (str): Name of configuration file for fitting the input-noise.
        stopat (int): Number of iterations.
        olhcmult (int): Scales the number of data points in the results files.

    Returns:
        None

    """
    
    #### check consistency
    datac, noisec = __read_file(data), __read_file(noise)
    datab, noiseb = __read_file(datac["beliefs"]), __read_file(noisec["beliefs"])
    if datac["inputs"] != noisec["inputs"]:
        print("\nWARNING: different inputs files in config files. Exiting.")
        return None 
    if datab["alt_nugget"] == 'F':
        print("\nWARNING: data beliefs must have alt_nugget T. Exiting.")
        return None
    if datab["fix_nugget"] == 'T' or  noiseb["fix_nugget"] == 'T':
        print("\nWARNING: data and noise beliefs need fix_nugget F. Exiting.")
        return None
    if datac["tv_config"] !=  noisec["tv_config"]:
        print("\nWARNING: different tv_config in config files. Exiting.")
        return None 
    if noisec["outputs"] != "zp-outputs":
        print("\nWARNING: config outputs file must be 'zp-outputs'. Exiting.")
        return None 

    ## setup emulators here
    GD = g.setup(data, datashuffle=True, scaleinputs=False)
    ## create 'zp-outputs' file with zeros
    #np.savetxt("zp-outputs", np.zeros(GD.training.outputs.size).T)
    np.savetxt("zp-outputs", np.zeros(GD.training.outputs.size + GD.validation.outputs.size).T)
    GN = g.setup(noise, datashuffle=True, scaleinputs=False)

    ## if shuffled, fix the inconsistencies
    GN.training.inputs = GD.training.inputs
    GN.validation.inputs = GD.validation.inputs
    GN.training.remake()
    GN.validation.remake()

    ## if we have validation sets, set no_retrain=True
    #if GD.all_data.tv.noV != 0:
    #    print("\nWARNING: need 0 validation sets in config files. Exiting.")
    #    exit()
    valsets = False if GD.all_data.tv.noV == 0 else True


    #### step 1 ####
    print("\n****************"
          "\nTRAIN GP ON DATA"
          "\n****************")
    #GD = g.setup(data, datashuffle=False, scaleinputs=False)
    x = GD.training.inputs # values of the inputs
    t = GD.training.outputs # values of the noisy outputs
    xv = GD.validation.inputs # values of the inputs
    tv = GD.validation.outputs

    #print(np.amin(x), np.amax(x))
    g.train(GD, no_retrain=valsets)

    r = np.zeros(t.size)
    rv = np.zeros(tv.size)
    ## we stay within this loop until done 'stopat' fits
    count = 0
    while True:
        if count == 0:
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
            xvp = __emuc.Data(xv, None, GD.basis, GD.par, GD.beliefs, GD.K)
        else:
            #### step 5 - return to step 2 if not converged ####
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
            xvp = __emuc.Data(xv, None, GD.basis, GD.par, GD.beliefs, GD.K)
            xp.set_r(r)
            xp.make_A(s2 = GD.par.sigma**2 , predict = True)
            xvp.set_r(rv)
            xvp.make_A(s2 = GD.par.sigma**2 , predict = True)
        count = count + 1


        #### step 2 - generate D'={(xi,zi)} ####
        print("\n***********************"
              "\nESTIMATING NOISE LEVELS " + str(count) +
              "\n***********************")

        post = __emuc.Posterior(xp, GD.training, GD.par, GD.beliefs, GD.K)
        L = np.linalg.cholesky(post.var)
        z_prime = np.zeros(t.size)
        s = samples
        for j in range(s): # predict 's' different values
            u = np.random.randn(t.size)
            tij = post.mean + L.dot(u)
            z_prime = z_prime + 0.5*(t - tij)**2
        z_prime = __transform(z_prime/float(s))
        np.savetxt('zp-outputs' , z_prime)

        # estimate noise levels for validation set
        post = __emuc.Posterior(xvp, GD.training, GD.par, GD.beliefs, GD.K)
        L = np.linalg.cholesky(post.var)
        z_prime_V = np.zeros(tv.size)
        s = samples
        for j in range(s): # predict 's' different values
            u = np.random.randn(tv.size)
            tij = post.mean + L.dot(u)
            z_prime_V = z_prime_V + 0.5*(tv - tij)**2
        z_prime_V = __transform(z_prime_V/float(s))
        #np.savetxt('zp-outputs' , z_prime)


        #### step 3 ####
        # train a GP on x and z
        print("\n*****************"
              "\nTRAIN GP ON NOISE " + str(count) +
              "\n*****************")
        ## need to setup again so as to re-read updated zp-outputs
        #GN = g.setup(noise, datashuffle=False, scaleinputs=False)
        #GN.training.outputs = np.loadtxt('zp-outputs').T
        GN.training.outputs = z_prime.T
        GN.training.remake()
        GN.validation.outputs = z_prime_V.T
        GN.validation.remake()
        ## fix to allow retraining using same training set against validation
        GN.tv_conf.no_of_trains = 0
        GN.tv_conf.retrain = 'y'
        g.train(GN, no_retrain=valsets)


        #### step 4 - use GN to predict noise values for G3 ####
        print("\n***********************************"
              "\nTRAIN GP ON DATA WITH NOISE FROM GP " + str(count) +
              "\n***********************************")

        xp_GN = __emuc.Data(x, None, GN.basis, GN.par, GN.beliefs, GN.K)
        p_GN = __emuc.Posterior(xp_GN, GN.training, GN.par, GN.beliefs, GN.K)
        r = __untransform(p_GN.mean)

        #GD = g.setup(data, datashuffle=False, scaleinputs=False)
        GD.training.set_r(r)

        ## add estimated r to the validation set for better diagnostics
        v_GN = __emuc.Data(xv, None, GN.basis, GN.par, GN.beliefs, GN.K)
        pv_GN = __emuc.Posterior(v_GN, GN.training, GN.par, GN.beliefs, GN.K)
        rv = __untransform(pv_GN.mean)
        GD.validation.set_r(rv)

        ## fix to allow retraining using same training set against validation
        GD.tv_conf.no_of_trains = 0
        GD.tv_conf.retrain = 'y'
        g.train(GD, no_retrain=valsets)

        # break when we've done 'stopat' fits
        if count == stopat:
            print("\nCompleted", count, "fits, stopping here.")

            ## use an OLHC design for x_values of noise guesses we'll save
            print("\nGenerating input points to predict noise values at...")
            n = x[0].size * int(olhcmult)
            N = int(n)
            olhc_range = [ [np.amin(col), np.amax(col)] for col in x.T ]
            #print("olhc_range:", olhc_range)
            filename = "x_range_input"
            _gd.optLatinHyperCube(x[0].size, n, N, olhc_range, filename)
            x_range = np.loadtxt(filename) # read generated oLHC file in

            # if 1D inputs, store in 2D array with only 1 column
            if x[0].size == 1:
                x_range = np.array([x_range,]).T

            ## save data to file
            x_plot = __emuc.Data(x_range, None, GN.basis, GN.par, GN.beliefs, GN.K)
            p_plot = __emuc.Posterior(x_plot, GN.training, GN.par, GN.beliefs, GN.K)
            mean_plot = p_plot.mean
            p_plot.interval()
            UI, LI = p_plot.UI, p_plot.LI
            print("\nSaving results to file...")
            np.savetxt('noise-inputs', x_range )
            np.savetxt('noise-outputs', np.transpose(\
              [np.sqrt(__untransform(mean_plot)),\
              np.sqrt(__untransform(LI)), np.sqrt(__untransform(UI))] ) )

            break

    return None

