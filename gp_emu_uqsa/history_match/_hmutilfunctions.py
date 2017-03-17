import gp_emu_uqsa.design_inputs as _gd
import gp_emu_uqsa._emulatorclasses as __emuc
import numpy as _np
import matplotlib.pyplot as _plt

def make_sets(ai):
    sets = [] # generate sets from active_index inputs
    for i in ai:
        for j in ai:
            if i!=j and i<j and [i,j] not in sets:
                sets.append([i,j])
    return sets


def emulsetup(emuls):
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
        sets = make_sets(ai)
        for i in range(len(ai)):
            #minmax[str(ai[i])] = mm[i]
            ## scale minmax into appropriate range
            minmax[str(ai[i])] = list( (_np.array(mm[i]) - mm[i][0])/(mm[i][1] - mm[i][0]) )
            orig_minmax[str(ai[i])] = list( (_np.array(mm[i])) )

    print("\nactive index pairs:" , sets)
    print("\nminmax for active inputs:" , minmax)
    print("original units minmax for active inputs:", orig_minmax)
    return sets, minmax, orig_minmax


def ref_act(minmax):
    ## reference active indices to ordered list of integers
    act_ref = {}
    count = 0
    for key in sorted(minmax.keys(), key=lambda x: int(x)):
        act_ref[key] = count
        count = count + 1
    print("\nrelate active_indices to integers:" , act_ref)
    return act_ref


def ref_plt(act):
    plt_ref = {}
    count = 0
    for key in sorted(act):
        plt_ref[str(key)] = count
        count = count + 1
    print("\nrelate restricted active_indices to subplot indices:" , plt_ref)
    return plt_ref


def check_act(act, sets):
    ## check 'act' is appropriate
    if type(act) is not list:
        print("ERROR: 'act' argument must be a list, but", act, "was supplied. Exiting.")
        exit()
    for a in act:
        if a not in [item for sublist in sets for item in sublist]:
            print("ERROR: index", a, "in 'act' is not an active_index of the emulator(s). Exiting.")
            exit()
    return True


def make_plots(s, plt_ref, cm, maxno, ax, IMP, ODP, minmax=None, recon=False):

    imp_pal = _plt.get_cmap('jet')
    odp_pal = _plt.get_cmap('afmhot')

    (odp, imp) = (ODP, IMP) if recon else (ODP[maxno-1], IMP[maxno-1])

    opd = _np.ma.masked_where(odp == 0, odp)
    ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].set_axis_bgcolor('darkgray')
 
    ex = None if recon else ( minmax[str(s[0])][0], minmax[str(s[0])][1],
                              minmax[str(s[1])][0], minmax[str(s[1])][1] )

    im_imp = ax[plt_ref[str(s[1])],plt_ref[str(s[0])]].imshow(imp.T,
      origin = 'lower', cmap = imp_pal, extent = ex,
      vmin=0.0, vmax=cm+1, interpolation='none' )
    im_odp = ax[plt_ref[str(s[0])],plt_ref[str(s[1])]].imshow(odp.T,
      origin = 'lower', cmap = odp_pal, extent = ex,
      vmin=0.0, vmax=1.0, interpolation='none' )

    _plt.colorbar(im_imp, ax=ax[plt_ref[str(s[1])],plt_ref[str(s[0])]])
    _plt.colorbar(im_odp, ax=ax[plt_ref[str(s[0])],plt_ref[str(s[1])]])
    return None


def plot_options(plt_ref, ax, fig, minmax=None):

    ## can set labels on diagaonal
    for key in plt_ref:
        ax[plt_ref[key],plt_ref[key]].set(adjustable='box-forced', aspect='equal')
        if minmax is not None:
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
    return None


def load_datafiles(datafiles, orig_minmax):

    try:
        sim_x, sim_y = _np.loadtxt(datafiles[0]), _np.loadtxt(datafiles[1])
    except FileNotFoundError as e:
        print("ERROR: datafile(s)", datafiles, "for inputs and/or outputs not found. Exiting.")
        exit()

    ## scale the inputs from the data file
    for key in orig_minmax.keys():
        sim_x[:,int(key)] = (sim_x[:,int(key)] - orig_minmax[key][0]) \
                              /(orig_minmax[key][1] - orig_minmax[key][0])

    return sim_x, sim_y
