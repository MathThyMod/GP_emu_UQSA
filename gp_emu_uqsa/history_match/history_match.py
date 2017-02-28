import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as __emuc
import numpy as _np
import matplotlib.pyplot as _plt


def imp(emuls, zs, cm, var_extra, grid=10, act=[]):

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
    for key in sorted(minmax.keys(), key=lambda x: x):
        act_ref[key] = count
        count = count + 1
    print("\nrelate active_indices to integers:" , act_ref)

    ## make a unifrom grid for variables {i,j}
    num_inputs = len(minmax) # number of inputs we'll look at
    dim = num_inputs - 2 # dimensions of input that we'll change with oLHC
    IMP, ODP = _np.zeros((grid,grid)) , _np.zeros((grid,grid))

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
        fig, ax = _plt.subplots(nrows = num_inputs, ncols = num_inputs)
        plt_ref = act_ref
    else:
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
        n = dim * 100  # no. of design_points - LET USER CHOOSE LATER
        N = n  # number of designs from which 1 maximin is chosen - LET USER CHOOSE LATER
        olhc_range = [it[1] for it in sorted(minmax.items(), key=lambda x: x[0]) \
                      if int(it[0])!=s[0] and int(it[0])!=s[1]]
        print("olhc_range:", olhc_range)
        filename = "imp_input"
        _gd.optLatinHyperCube(dim, n, N, olhc_range, filename)
        x_other_inputs = _np.loadtxt(filename) # read generated oLHC file in
        
        ## enough for ALL inputs - we'll mask any inputs not used by a particular emulator later
        x = _np.empty( [n , num_inputs] )

        ###################################################
        ## stepping over the grid {i,j} to build subplot ##
        ###################################################
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
                        #print("act_ind_list:" , act_ind_list)
                        ni = __emuc.Data(x[:,act_ind_list],None,E.basis,E.par,E.beliefs,E.K)
                        post = __emuc.Posterior(ni, E.training, E.par, E.beliefs, E.K, predict=True)
                        mean = post.mean
                        var  = _np.diag(post.var)

                        ## calculate implausibility^2 values
                        for r in range(0,n):
                            I2[r,o] = ( mean[r] - z )**2 / ( var[r] + var_e )

                ## find maximum implausibility across different outputs
                I = _np.sqrt(I2)
                odp_count = 0
                for r in range(0,n):
                    I[r,0] = _np.amax( I[r,:] ) # place maximum in first column
                    if I[r,0] < cm: # check cut-off using this value
                        odp_count = odp_count + 1

                ## then find the minimum of those implausibilities across the n points
                IMP[i,j] = _np.amin(I[:,0]) # must only use fisrt column
                ## make the optical depth plots after having looped over the emulators
                ODP[i,j] = float(odp_count) / float(n)


        ## save the results to file
        _np.savetxt("IMP_"+str(s[0])+'_'+str(s[1]), IMP)
        _np.savetxt("ODP_"+str(s[0])+'_'+str(s[1]), ODP)

        ## minimum implausibility 
        #imp_pal = _plt.get_cmap('viridis_r')
        imp_pal = _plt.get_cmap('jet')
        im = ax[plt_ref[str(s[1])],plt_ref[str(s[0])]].imshow(IMP.T, origin = 'lower', cmap = imp_pal,
          extent = ( minmax[str(s[0])][0], minmax[str(s[0])][1],
                     minmax[str(s[1])][0], minmax[str(s[1])][1]),
          vmin=0.0, vmax=cm + 1,
          interpolation='none' )

        ## optical depth plot
        #odp_pal = _plt.get_cmap('plasma_r')
        odp_pal = _plt.get_cmap('afmhot')
        ODP = _np.ma.masked_where(ODP == 0, ODP)
        ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].set_axis_bgcolor('darkgray')
        m2 = ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].imshow(ODP.T,
          origin = 'lower', cmap = odp_pal,
          extent = ( minmax[str(s[0])][0], minmax[str(s[0])][1],
                     minmax[str(s[1])][0], minmax[str(s[1])][1]),
          vmin=0.0, vmax=1.0,
          interpolation='none' )

    ###############################
    ## sort out the overall plot ##
    ###############################

    ## can set labels on diagaonal
    for key in plt_ref:
        ax[plt_ref[key],plt_ref[key]].set(adjustable='box-forced', aspect='equal')
        ax[plt_ref[key],plt_ref[key]].text(.25,.5,"Input " + str(key) + "\n"
           + str(minmax[key][0]) + "\n-\n" + str(minmax[key][1]))
        #fig.delaxes(ax[a,a]) # for deleting the diagonals

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


