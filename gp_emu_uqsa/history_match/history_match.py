import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as __emuc
import numpy as _np
import matplotlib.pyplot as _plt


def imp(emuls, zs, cm, var_extra, maxno=1, olhcmult=100, grid=10, act=[], fileStr="", plot=True):

    if plot != True:
        plot = False

    maxno=int(maxno)

    sets = [] # generate sets from active_index inputs
    minmax = {} # fetch minmax information from the beliefs files
    for e in emuls:
        try:
            ai = e.beliefs.active_index
            mm = e.beliefs.input_minmax
        except AttributeError as e:
            print("ERROR: Emulator(s) were not previously trained and reconstructed "
                  "using updated beliefs files, "
                  "so they are missing 'active_index' and 'input_minmax'. Exiting.")
            exit()
        for i in ai:
            for j in ai:
                if i!=j and i<j and [i,j] not in sets:
                    sets.append([i,j])
        for i in range(len(ai)):
            #minmax[str(ai[i])] = mm[i]
            ## scale minmax into appropriate range
            minmax[str(ai[i])] = list( (_np.array(mm[i]) - mm[i][0])/(mm[i][1] - mm[i][0]) )
    print("\nactive index pairs:" , sets)
    print("\nminmax for active inputs:" , minmax)

    ## reference active indices to ordered list of integers
    act_ref = {}
    count = 0
    for key in sorted(minmax.keys(), key=lambda x: int(x)):
        act_ref[key] = count
        count = count + 1
    print("\nrelate active_indices to integers:" , act_ref)

    ## make a unifrom grid for variables {i,j}
    num_inputs = len(minmax) # number of inputs we'll look at
    dim = num_inputs - 2 # dimensions of input that we'll change with oLHC

    ## need enough IMP and ODP for each I_max (e.g. I_first_max, I_second_max) that we need
    IMP , ODP = [], []
    for i in range(maxno):
        IMP.append( _np.zeros((grid,grid)) )
        ODP.append( _np.zeros((grid,grid)) )

    ## check 'act' is appropriate
    if type(act) is not list:
        print("ERROR: 'act' argument must be a list, but", act, "was supplied. Exiting.")
        exit()
    for a in act:
        if a not in [item for sublist in sets for item in sublist]:
            print("ERROR: index", a, "in 'act' is not an active_index of the emulator(s). Exiting.")
            exit()

    ## space for all plots, and reference index to subplot indices
    if act == []:
        if plot:
            fig, ax = _plt.subplots(nrows = num_inputs, ncols = num_inputs)
        plt_ref = act_ref
    else:
        if plot:
            fig, ax = _plt.subplots(nrows = len(act), ncols = len(act))
        plt_ref = {}
        count = 0
        for key in sorted(act):
            plt_ref[str(key)] = count
            count = count + 1
        print("\nrelate restricted active_indices to subplot indices:" , plt_ref)

    ## reduce sets to only the chosen ones
    less_sets = []
    if act == []:
        less_sets = sets
    else:
        for s in sets:
            if s[0] in act and s[1] in act:
                less_sets.append(s)

    print("HM for input pairs:", less_sets)

    ############################################
    ## calculate plot for each pair of inputs ##
    ############################################
    for s in less_sets:
        print("\nset:", s)

        ## rows and columns of 2D grid for the {i,j} value of pair of inputs
        X1 = _np.linspace(minmax[str(s[0])][0], minmax[str(s[0])][1], grid, endpoint=False)
        X1 = X1 + 0.5*(minmax[str(s[0])][1] - minmax[str(s[0])][0])/float(grid)
        X2 = _np.linspace(minmax[str(s[1])][0], minmax[str(s[1])][1], grid, endpoint=False)
        X2 = X2 + 0.5*(minmax[str(s[1])][1] - minmax[str(s[1])][0])/float(grid)
        print("Values of the grid 1:" , X1)
        print("Values of the grid 2:" , X2)
        x_all=_np.zeros((grid*grid,2))
        for i in range(0,grid):
            for j in range(0,grid):
                x_all[i*grid+j,0] = X1[i]
                x_all[i*grid+j,1] = X2[j]


        ## use an OLHC design for all remaining inputs
        n = dim * int(olhcmult)  # no. of design_points - LET USER CHOOSE LATER
        N = int(n/2)  # number of designs from which 1 maximin is chosen - LET USER CHOOSE LATER
        olhc_range = [it[1] for it in sorted(minmax.items(), key=lambda x: int(x[0])) \
                      if int(it[0])!=s[0] and int(it[0])!=s[1]]
        print("olhc_range:", olhc_range)
        filename = "imp_input_"+str(s[0])+'_'+str(s[1])
        _gd.optLatinHyperCube(dim, n, N, olhc_range, filename)
        x_other_inputs = _np.loadtxt(filename) # read generated oLHC file in
        
        ## enough for ALL inputs - we'll mask any inputs not used by a particular emulator later
        x = _np.empty( [n , num_inputs] )

        ###################################################
        ## stepping over the grid {i,j} to build subplot ##
        ###################################################
        print("\nCalculating Implausibilities...")
        for i in range(0,grid):
            for j in range(0,grid):
                
                ## create array to store the implausibility values for the x values
                I2 = _np.zeros((n,len(emuls)))

                ## loop over outputs (i.e. over emulators)
                for o in range(len(emuls)):
                    #print("\n*** Emulator:", o ,"***")
                    E, z, var_e = emuls[o], zs[o], var_extra[o]

                    ## check if these inputs are active for this emulator
                    Eai = E.beliefs.active_index
                    ind_in_active=True if s[0] in Eai and s[1] in Eai else False
                    if ind_in_active:

                        ## set the input pair for this subplot
                        x[:,act_ref[str(s[0])]] = x_all[i*grid+j, 0]
                        x[:,act_ref[str(s[1])]] = x_all[i*grid+j, 1]

                        ## figure out what the other inputs active_indices are
                        other_dim = [act_ref[str(key)] for key in act_ref if int(key) not in s]
                        #print("other dim:" , other_dim)
                        if len(other_dim) == 1:
                            x[:,other_dim] = _np.array([x_other_inputs,]).T
                        else:
                            x[:,other_dim] = x_other_inputs
                        
                        ## inactive inputs are masked
                        act_ind_list = [act_ref[str(l)] for l in Eai]
                        #print("active_indices for emulator", o, "are:", act_ind_list)
                        ni = __emuc.Data(x[:,act_ind_list],None,E.basis,E.par,E.beliefs,E.K)
                        post = __emuc.Posterior(ni, E.training, E.par, E.beliefs, E.K, predict=True)
                        mean = post.mean
                        var  = _np.diag(post.var)

                        ## calculate implausibility^2 values
                        for r in range(0,n):
                            I2[r,o] = ( mean[r] - z )**2 / ( var[r] + var_e )

                ## find maximum implausibility across different outputs
                I = _np.sqrt(I2)
                odp_count = _np.zeros(maxno,dtype=_np.uint32)
                Imaxes = _np.empty([n,maxno])
                for r in range(0,n):
                    #I[r,0] = _np.amax( I[r,:] ) # place maximum in first column
                    # 'Imaxes' stores the maxes in ascending order e.g. [10,11,12]
                    Imaxes[r,:] = _np.sort(_np.partition(I[r,:],-maxno)[-maxno:])[-maxno:]
                    #I[r,0] = _np.sort(_np.partition(I[r,:],-int(maxno))[-int(maxno):])[-int(maxno)]

                    for m in range(maxno):
                        if Imaxes[r,-(m+1)] < cm: # check cut-off using this value
                            odp_count[m] = odp_count[m] + 1

                for m in range(maxno):
                    ## then find the minimum of those implausibilities across the n points
                    IMP[m][i,j] = _np.amin(Imaxes[:,-(m+1)]) # access from end of array first
                    ## make the optical depth plots after having looped over the emulators
                    ODP[m][i,j] = float(odp_count[m]) / float(n)


        ## save the results to file
        if fileStr != "":
            nfileStr = fileStr + "_"
        else:
            nfileStr = fileStr
        ## different file for each max
        for m in range(maxno):
            _np.savetxt(nfileStr+str(m+1)+"_"+"IMP_"+str(s[0])+'_'+str(s[1]), IMP[m])
            _np.savetxt(nfileStr+str(m+1)+"_"+"ODP_"+str(s[0])+'_'+str(s[1]), ODP[m])

        ## minimum implausibility 
        #imp_pal = _plt.get_cmap('viridis_r')

        if plot:

            imp_pal = _plt.get_cmap('jet')
            im = ax[plt_ref[str(s[1])],plt_ref[str(s[0])]].imshow(IMP[maxno-1].T,
              origin = 'lower', cmap = imp_pal,
              extent = ( minmax[str(s[0])][0], minmax[str(s[0])][1],
                         minmax[str(s[1])][0], minmax[str(s[1])][1]),
              vmin=0.0, vmax=cm + 1,
              interpolation='none' )
            _plt.colorbar(im, ax=ax[plt_ref[str(s[1])],plt_ref[str(s[0])]])

            ## optical depth plot
            #odp_pal = _plt.get_cmap('plasma_r')
            odp_pal = _plt.get_cmap('afmhot')
            ODP[maxno-1] = _np.ma.masked_where(ODP[maxno-1] == 0, ODP[maxno-1])
            ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].set_axis_bgcolor('darkgray')
            m2 = ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].imshow(ODP[maxno-1].T,
              origin = 'lower', cmap = odp_pal,
              extent = ( minmax[str(s[0])][0], minmax[str(s[0])][1],
                         minmax[str(s[1])][0], minmax[str(s[1])][1]),
              vmin=0.0, vmax=1.0,
              interpolation='none' )
            _plt.colorbar(m2, ax=ax[plt_ref[str(s[0])],plt_ref[str(s[1])]])


    ###############################
    ## sort out the overall plot ##
    ###############################

    if plot:

        ## can set labels on diagaonal
        for key in plt_ref:
            ax[plt_ref[key],plt_ref[key]].set(adjustable='box-forced', aspect='equal')
            ax[plt_ref[key],plt_ref[key]].text(.25,.5,"Input " + str(key) + "\n"
               + str(minmax[key][0]) + "\n-\n" + str(minmax[key][1]))
            fig.delaxes(ax[plt_ref[key],plt_ref[key]]) # for deleting the diagonals

        ## can remove ticks using something like this    
        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect('equal')

        ## set the ticks on the edges
        #for i in range(len(minmax)):
        #    for j in range(len(minmax)):
        #        if i != len(minmax) - 1:
        #            ax[i,j].set_xticks([])
        #        if j != 0:
        #            ax[i,j].set_yticks([])
            

        _plt.tight_layout()
        _plt.show()

    return


def imp_recon(ai,fileStr,cm,maxno=1):

    sets=[]
    for i in ai:
        for j in ai:
            if i!=j and i<j and [i,j] not in sets:
                sets.append([i,j])

    act = ai
    fig, ax = _plt.subplots(nrows = len(act), ncols = len(act))
    plt_ref = {}
    count = 0
    for key in sorted(act):
        plt_ref[str(key)] = count
        count = count + 1
    print("\nrelate restricted active_indices to subplot indices:" , plt_ref)


    print("HM for input pairs:", sets)
    ############################################
    ## calculate plot for each pair of inputs ##
    ############################################
    for s in sets:
        print("\nset:", s)

        ## save the results to file
        if fileStr != "":
            nfileStr = fileStr + "_"
        else:
            nfileStr = fileStr
        IMP = _np.loadtxt(nfileStr+str(maxno)+"_"+"IMP_"+str(s[0])+'_'+str(s[1]))
        ODP = _np.loadtxt(nfileStr+str(maxno)+"_"+"ODP_"+str(s[0])+'_'+str(s[1]))

        ## minimum implausibility 
        #imp_pal = _plt.get_cmap('viridis_r')
        imp_pal = _plt.get_cmap('jet')
        im = ax[plt_ref[str(s[1])],plt_ref[str(s[0])]].imshow(IMP.T, origin = 'lower', cmap = imp_pal,
          vmin=0.0, vmax=cm + 1,
          interpolation='none' )
        _plt.colorbar(im, ax=ax[plt_ref[str(s[1])],plt_ref[str(s[0])]])

        ## optical depth plot
        #odp_pal = _plt.get_cmap('plasma_r')
        odp_pal = _plt.get_cmap('afmhot')
        ODP = _np.ma.masked_where(ODP == 0, ODP)
        ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].set_axis_bgcolor('darkgray')
        m2 = ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].imshow(ODP.T,
          origin = 'lower', cmap = odp_pal,
          vmin=0.0, vmax=1.0,
          interpolation='none' )
        _plt.colorbar(m2, ax=ax[plt_ref[str(s[0])],plt_ref[str(s[1])]])


    ###############################
    ## sort out the overall plot ##
    ###############################

    ## can set labels on diagaonal
    for key in plt_ref:
        ax[plt_ref[key],plt_ref[key]].set(adjustable='box-forced', aspect='equal')
        #ax[plt_ref[key],plt_ref[key]].text(.25,.5,"Input " + str(key) + "\n"
        #   + str(minmax[key][0]) + "\n-\n" + str(minmax[key][1]))
        fig.delaxes(ax[plt_ref[key],plt_ref[key]]) # for deleting the diagonals

    ## can remove ticks using something like this    
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect('equal')

    ## set the ticks on the edges
    #for i in range(len(minmax)):
    #    for j in range(len(minmax)):
    #        if i != len(minmax) - 1:
    #            ax[i,j].set_xticks([])
    #        if j != 0:
    #            ax[i,j].set_yticks([])
        

    _plt.tight_layout()
    _plt.show()

    return


def non_imp(emuls, zs, cm, var_extra, datafiles, maxno=1, act=[], fileStr=""):

    ## enough for ALL inputs - we'll mask any inputs not used by a particular emulator later
    #x = _np.empty( [n , num_inputs] )
    try:
        sim_x = _np.loadtxt(datafiles[0])
        sim_y = _np.loadtxt(datafiles[1])
    except FileNotFoundError as e:
        print("ERROR: datafile(s)", datafiles, "for inputs and/or outputs not found. Exiting.")
        exit()

    ## NEED TO SCALE the inputs based on minmax!
    maxno=int(maxno)

    minmax = {} # fetch minmax information from the beliefs files
    orig_minmax = {} # fetch minmax information from the beliefs files
    for e in emuls:
        try:
            ai = e.beliefs.active_index
            mm = e.beliefs.input_minmax
        except AttributeError as e:
            print("ERROR: Emulator(s) were not previously trained and reconstructed "
                  "using updated beliefs files, "
                  "so they are missing 'active_index' and 'input_minmax'. Exiting.")
            exit()
        for i in range(len(ai)):
            #minmax[str(ai[i])] = mm[i]
            ## scale minmax into appropriate range
            minmax[str(ai[i])] = list( (_np.array(mm[i]) - mm[i][0])/(mm[i][1] - mm[i][0]) )
            orig_minmax[str(ai[i])] = list( (_np.array(mm[i])) )
    print("\nminmax for active inputs:" , minmax)
    print("original units minmax for active inputs:", orig_minmax)
    
    for key in orig_minmax.keys():
        sim_x[:,int(key)] = (sim_x[:,int(key)] - orig_minmax[key][0]) \
                              /(orig_minmax[key][1] - orig_minmax[key][0])

    ## reference active indices to ordered list of integers
    act_ref = {}
    count = 0
    for key in sorted(minmax.keys(), key=lambda x: int(x)):
        act_ref[key] = count
        count = count + 1
    print("\nrelate active_indices to integers:" , act_ref)

    ## make a unifrom grid for variables {i,j}
    num_inputs = len(minmax) # number of inputs we'll look at

    ## check 'act' is appropriate
    if type(act) is not list:
        print("ERROR: 'act' argument must be a list, but", act, "was supplied. Exiting.")
        exit()
    for a in act:
        if a not in ai: ## THIS IS MODIFIED FOR THIS FUNCTION
            print("ERROR: index", a, "in 'act' is not an active_index of the emulator(s). Exiting.")
            exit()


    ##################################
    ## see which inputs are non-imp ##
    ## for the sim results we have  ##
    ##################################
 

    ## FOR NOW, I'LL ASSUME WE'VE SUPPLIED A FILE THAT CONTAINS *ALL* INPUTS
    ## THESE ARE THE POINTS WE'LL BE TESTING FOR NON-IMP
    ## ALSO A DATAFILE CONTAINING ALL OUTPUTS SO THAT WE CAN PRODUCE NEW INPUT AND OUTPUT
    ## FILES FOR THE NEW WAVE


    # 'n' must become how many input points we have
    n = sim_x[:,0].size

    print("\nCalculating Implausibilities...")
    ## create array to store the implausibility values for the x values
    I2 = _np.zeros((n,len(emuls)))

    ## loop over outputs (i.e. over emulators)
    for o in range(len(emuls)):
        #print("\n*** Emulator:", o ,"***")
        E, z, var_e = emuls[o], zs[o], var_extra[o]

        Eai = E.beliefs.active_index

        ## inactive inputs are masked
        act_ind_list = [act_ref[str(l)] for l in Eai]

        #print("active_indices for emulator", o, "are:", act_ind_list)
        ni = __emuc.Data(sim_x[:,act_ind_list],None,E.basis,E.par,E.beliefs,E.K)
        post = __emuc.Posterior(ni, E.training, E.par, E.beliefs, E.K, predict=True)
        mean = post.mean
        var  = _np.diag(post.var)

        ## calculate implausibility^2 values
        for r in range(0,n):
            I2[r,o] = ( mean[r] - z )**2 / ( var[r] + var_e )

    ## find maximum implausibility across different outputs
    I = _np.sqrt(I2)
    Imaxes = _np.empty([n,maxno])
    ## for saving non-imp values
    nimp_inputs = []
    nimp_outputs = []
    for r in range(0,n):
        Imaxes[r,:] = _np.sort(_np.partition(I[r,:],-maxno)[-maxno:])[-maxno:]

        # make a loop later on fro different maxex... maybe... maybe not
        #for m in range(maxno):
        m = maxno-1
        if Imaxes[r,-(m+1)] < cm: # check cut-off using this value
            nimp_inputs.append(sim_x[r,:])
            nimp_outputs.append(sim_y[r,:])

    ## save the results to file
    if fileStr != "":
        nfileStr = fileStr + "_"
    else:
        nfileStr = fileStr
    ## different file for each max
    for m in range(maxno):
        #_np.savetxt(nfileStr+str(m+1)+"_"+datafiles[0], nimp_inputs)
        #_np.savetxt(nfileStr+str(m+1)+"_"+datafiles[1], nimp_outputs)
        _np.savetxt(nfileStr + datafiles[0], nimp_inputs)
        _np.savetxt(nfileStr + datafiles[1], nimp_outputs)

    return
