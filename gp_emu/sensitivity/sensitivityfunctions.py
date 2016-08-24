### for the user accessable sensitivity functions
import numpy as _np
import matplotlib.pyplot as plt
from ._sensitivityclasses import *


def setup(emul,  m, v, case="case2"):
    #print("\nsetup function for initialising Sensitivity class")
    m = _np.array(m)
    v = _np.array(v)
    s = Sensitivity(emul, m, v)
    return s


def sense_table(sense_list, inputNames, outputNames, rowHeight=6):
    rows = len(sense_list)
    cols = len(sense_list[0].m) + 1
    if inputNames == []:
        inputNames = ["input " + str(i) for i in range(cols-1)]
    inputNames.append("E*[var[f(X)]]")
    print(inputNames)
    if outputNames == []:
        outputNames = ["output " + str(i) for i in range(rows)]
    #print("rows X cols:" , rows , "X" , cols)
    cells = np.zeros([rows, cols])
    #cells_col = np.zeros([rows, cols])
    si = 0
    for s in sense_list:
        #print("Inside loop over sensitivity instances")
        cells[si,0:cols-1] = s.senseindex/s.uEV
        cells[si,cols-1] = np.sum(s.senseindex/s.uEV)
        #cells_col[si,0:cols-1] = s.senseindex/s.uEV
        #cells_col[si,cols-1] = 1.0
        #print(cells[si])
        si = si + 1

    #### create the sensitivity table
    #fig = plt.figure(figsize=(8,4))
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111, frameon=False, xticks = [], yticks = [])
    #img = plt.imshow(cells, cmap="hot")
    img = plt.imshow(cells, cmap="hot", vmin=0.0, vmax=1.0)
    #plt.colorbar()
    img.set_visible(False)
    tb = plt.table(cellText = cells, 
        colLabels = inputNames, 
        rowLabels = outputNames,
        loc = 'center',
        cellColours = img.to_rgba(cells))
        #cellColours = img.to_rgba(cells_col))
    #tb.set_fontsize(34)
    tb.scale(1,rowHeight)
    for i in range(1, rows+1):
        for j in range(0, cols):
            tb._cells[(i,j)]._text.set_color('green')
    # ax.add_table(tb)
    plt.show()

