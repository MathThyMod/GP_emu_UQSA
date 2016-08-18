### for the user accessable sensitivity functions
import numpy as _np
import matplotlib.pyplot as plt
from ._sensitivityclasses import *

def setup(emul, case, m, v):
    print("\nsetup function for initialising Sensitivity class")
    m = _np.array(m)
    v = _np.array(v)
    s = Sensitivity(emul, m, v)
    return s


def sense_table(sense_list, inputNames, outputNames):
    rows = len(sense_list)
    cols = len(sense_list[0].m)
    if inputNames == []:
        inputNames = ["input " + str(i) for i in range(cols)]
    if outputNames == []:
        outputNames = ["output " + str(i) for i in range(rows)]
    print("rows X cols:" , rows , "X" , cols)
    cells = np.zeros([rows, cols])
    si = 0
    for s in sense_list:
        print("Inside loop over sensitivity instances")
        cells[si] = s.senseindex/s.uEV
        print(cells[si])
        si = si + 1

    #### create the sensitivity table
    #fig = plt.figure(figsize=(8,4))
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])
    img = plt.imshow(cells, cmap="hot")
    #plt.colorbar()
    img.set_visible(False)
    tb = plt.table(cellText = cells, 
        colLabels = inputNames, 
        rowLabels = outputNames,
        loc = 'center',
        cellColours = img.to_rgba(cells))
    #tb.set_fontsize(34)
    tb.scale(1,2)
    for i in range(1, rows+1):
        for j in range(0, cols):
            tb._cells[(i,j)]._text.set_color('green')
    # ax.add_table(tb)
    plt.show()
