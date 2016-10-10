### for the user accessable sensitivity functions
import numpy as _np
import matplotlib.pyplot as plt
from ._sensitivityclasses import *


def setup(emul,  m, v, case="case2"):
    '''
    Return an instance of the sensitivity class initialised with m and v.

    Args:
        emul (Emulator): Instance of the emulator class.
        m (list of floats): Means of the input dimensions
        v (list of floats): Variances of the input dimensions.
        case (str): Which routines to use. Must be "case2" (default).

    Returns:
        Sensitivity: Initialised Sensitivity class.
    '''

    print("\n*** Initialising Sensitivity class ***")

    # check that m and v are actually
    try:
        assert isinstance(m , list)
        assert isinstance(v , list)
    except AssertionError as e:
        print("ERROR: 2nd and 3rd arguments must be lists of floats. "
              "Return None.")
        return None

    if case=="case2":
        # make sure kernel is a Gaussian 
        if len(emul.K.name)>1 \
          or (emul.K.name[0] != "gaussian_mucm" \
              and emul.K.name[0] != "gaussian"):
            print("The case2 sensitivity routines only work for emulators "
                  "with a gaussian kernel and linear mean. "
                  "This emulator Kernel will not work. Return None.")
            return None
        # make sure mean function is linear
        if len(emul.par.beta) != emul.training.inputs[0].size+1 \
          or False in [i=='x' for i in emul.beliefs.basis_str[1:]]:
            print("The case2 sensitivity routines only work for emulators "
                  "with a gaussian kernel and linear mean. "
                  "This mean function will not work. Return None.")
            return None
        # ensure that supplied m and v lists are long enough
        if len(m) != len(emul.par.beta) - 1 \
          or len(v) != len(emul.par.beta) - 1:
            print("Mean and Variance lists must both contain as many items "
                  "as there are input dimensions. Return None.")
            return None
    else:
        print("Only case2 of MUCM's U & S analysis is implemented. Return None.")
        return None


    m = _np.array(m)
    v = _np.array(v)
    s = Sensitivity(emul, m, v)
    return s


def sense_table(sense_list, inputNames=[], outputNames=[], rowHeight=6):
    '''
    Create a table plot of sensitivity indices. Rows are sensitivity instances (presumably initialised with different trained emulators), columns are input dimensions. Example use: build a different emulator for every output using all inputs and plot a table showing the sensitivity of each output to each input. Return None.

    Args:
        sense_list (list(Sensitivity)): List of Sensitivity instances.
        inputNames (list(str)): Optional list of column titles.
        outputNames (list(str)): Optional list of row titles.
        rowHeight (float): Optional float to scale row height (default 6).

    Returns:
        None.

    '''

    # make sure the input is a list
    try:
        assert isinstance(sense_list , list)
    except AssertionError as e:
        print("ERROR: first argument must be list e.g. [s] or [s,] or [s1, s2]. "
              "Return None.")
        return None

    rows = len(sense_list)
    cols = len(sense_list[0].m) + 1

    # ensure same number of inputs for each emulator
    for s in sense_list:
        if len(s.m) != cols - 1:
            print("Each emulator must be built with the same number of inputs.")
            return None

    # if required routines haven't been run yet, then run them
    for s in sense_list:
        if s.done_uncertainty == False:
            s.uncertainty()
        if s.done_sensitivity == False:
            s.sensitivity()

    # name the rows and columns
    if inputNames == []:
        inputNames = ["input " + str(i) for i in range(cols-1)]
    inputNames.append("Sum")
    if outputNames == []:
        outputNames = ["output " + str(i) for i in range(rows)]

    cells = np.zeros([rows, cols])
    # iterate over rows of the sense_table
    si = 0
    for s in sense_list:
        cells[si,0:cols-1] = s.senseindex/s.uEV
        cells[si,cols-1] = np.sum(s.senseindex/s.uEV)
        si = si + 1
    
    # a way to format the numbers in the table
    tab_2 = [['%.3f' % j for j in i] for i in cells]

    ## create the sensitivity table

    # table size
    #fig = plt.figure(figsize=(8,4))
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])

    # table color and colorbar
    #img = plt.imshow(cells, cmap="hot")
    img = plt.imshow(cells, cmap="hot", vmin=0.0, vmax=1.0)
    #plt.colorbar()
    img.set_visible(False)
    
    # create table
    tb = plt.table(cellText = tab_2, 
        colLabels = inputNames, 
        rowLabels = outputNames,
        loc = 'center',
        cellColours = img.to_rgba(cells))
        #cellColours = img.to_rgba(cells_col))

    # fix row height and text
    #tb.set_fontsize(34)
    tb.scale(1,rowHeight)

    # change text color to make more visible
    for i in range(1, rows+1):
        for j in range(0, cols):
            tb._cells[(i,j)]._text.set_color('green')

    plt.show()

    return None
