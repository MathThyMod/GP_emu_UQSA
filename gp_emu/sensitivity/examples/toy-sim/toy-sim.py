import sys
import numpy as np

def sim2D(x):
    dim = x[:,0].size
    print("input dim:", dim)
    if dim == 1:
        y = 3.0*x[0]**3
    if dim == 2:
        y = 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2)
    return y


if len(sys.argv)>1:
    noise=float(sys.argv[1])
    print("noise set to:",noise)
else:
    noise=0.0
    print("noise not set, setting to:",noise)

inputfile="toy-sim_input"
print("Looking for input file:", inputfile)
inputs=(np.loadtxt(inputfile)).transpose()
#print("inputs:\n", inputs)

y=sim2D(inputs)
y = y + noise*np.random.randn(y.size)

outputfile="toy-sim_output"
print("outputs to output file:", outputfile)
np.savetxt(outputfile, y, delimiter=' ', fmt='%1.4f')

