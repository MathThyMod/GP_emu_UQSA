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

### read files
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
def noisefit(data, noise, stopat=20):
    
    ## NOTES
    #### currently no_retrain is False so we would always train to final set...
    #### if we left a validation set out, problem is that noise variance r would not be included...
    ####### could fix with GN.posterior for the validation points and using g3.validation.set_r() ?
    #### I SHOULD MAKE USE OF THE NEW POSTERIOR AND POSTERIOS SAMPLE ROUTINES
    #### I need to generalize the saved results - make two files


    #### setup up emulators to check consistency
    datac, noisec = __read_file(data), __read_file(noise)
    datab, noiseb = __read_file(datac["beliefs"]), __read_file(noisec["beliefs"])
    if datac["inputs"] != noisec["inputs"]:
        print("\nWARNING: different inputs files referred to in the supplied config files. Exiting.")
        return None 
    if datab["alt_nugget"] == 'F':
        print("\nWARNING: data beliefs must have alt_nugget T. Exiting.")
        return None
    if datab["fix_nugget"] == 'T' or  noiseb["fix_nugget"] == 'T':
        print("\nWARNING: data and noise beliefs must have fix_nugget F. Exiting.")
        return None


    #### step 1 ####
    print("\n****************"
          "\nTRAIN GP ON DATA"
          "\n****************")
    GD = g.setup(data, datashuffle=False, scaleinputs=False)
    x = GD.training.inputs # values of the inputs
    t = GD.training.outputs # values of the noisy outputs

    #x_range = np.array( (np.linspace(np.amin(x), np.amax(x), t.size),) ).T
    ## use an OLHC design for x_values of noise guesses we'll save
    n = x[0].size * 100 
    N = int(n/2)
    olhc_range = [ [np.amin(col), np.amax(col)] for col in x.T ]
    #print("olhc_range:", olhc_range)
    filename = "x_range_input"
    _gd.optLatinHyperCube(x[0].size, n, N, olhc_range, filename)
    x_range = np.loadtxt(filename) # read generated oLHC file in

    #print(np.amin(x), np.amax(x))
    g.train(GD, no_retrain=False)

    r = np.zeros(t.size)
    ## we stay within this loop until done 'stopat' fits
    count = 0
    while True:
        if count == 0:
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
        else:
            #### step 5 - return to step 2 if not converged ####
            xp = __emuc.Data(x, None, GD.basis, GD.par, GD.beliefs, GD.K)
            xp.set_r(r)
            xp.make_A(s2 = GD.par.sigma**2 , predict = True)
        count = count + 1


        #### step 2 - generate D'={(xi,zi)} ####
        print("\n***********************"
              "\nESTIMATING NOISE LEVELS " + str(count) +
              "\n***********************")

        post = __emuc.Posterior(xp, GD.training, GD.par, GD.beliefs, GD.K)
        L = np.linalg.cholesky(post.var)
        z_prime = np.zeros(t.size)
        s = 200
        for j in range(s): # predict 's' different values
            u = np.random.randn(t.size)
            tij = post.mean + L.dot(u)
            z_prime = z_prime + 0.5*(t - tij)**2
        z_prime = __transform(z_prime/float(s))
        np.savetxt('zp-outputs' , z_prime)


        #### step 3 ####
        # train a GP on x and z
        print("\n*****************"
              "\nTRAIN GP ON NOISE " + str(count) +
              "\n*****************")
        GN = g.setup(noise, datashuffle=False, scaleinputs=False)
        g.train(GN, no_retrain=False)


        #### step 4 - use GN to predict noise values for G3 ####
        print("\n***********************************"
              "\nTRAIN GP ON DATA WITH NOISE FROM GP " + str(count) +
              "\n***********************************")

        xp_GN = __emuc.Data(x, None, GN.basis, GN.par, GN.beliefs, GN.K)
        p_GN = __emuc.Posterior(xp_GN, GN.training, GN.par, GN.beliefs, GN.K)
        r = __untransform(p_GN.mean)

        GD = g.setup(data, datashuffle=False, scaleinputs=False)
        GD.training.set_r(r)
        g.train(GD, no_retrain=False)

        ## save data to file
        x_plot = __emuc.Data(x_range, None, GN.basis, GN.par, GN.beliefs, GN.K)
        post_plot = __emuc.Posterior(x_plot, GN.training, GN.par, GN.beliefs, GN.K)
        mean_plot = post_plot.mean
        post_plot.interval()
        UI, LI = post_plot.UI, post_plot.LI
        print("\nSaving results to file...")
        np.savetxt('noise-inputs', x_range )
          #[ mean_plot, np.sqrt(__untransform(mean_plot)),\
          #np.sqrt(__untransform(LI)), np.sqrt(__untransform(UI)) ] )
        np.savetxt('noise-outputs', mean_plot )

        # break when we've done 'stopat' fits
        if count == stopat:
            print("\nCompleted", count, "fits, stopping here.")
            break

    return None

