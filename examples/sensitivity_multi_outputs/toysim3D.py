import sys
import numpy as np

def sim(x):
    dim = x[:,0].size
    y = np.zeros([x[0,:].size,2])
    print("input dim:", dim)
    if dim == 1:
        y[0] = 3.0*x[0]**3
        y[1] = 2.0*x[0]**2
    if dim == 2:
        y[0] = 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2)
        y[1] = 2.0*x[0]**2 + np.exp(np.cos(10.0*x[0])*np.cos(5.0*x[1])**2)
    if dim == 3:
        print(x[0],x[1],x[2])
        y[:,0] = 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2) + np.exp(np.sin(7.5*x[2]))
        y[:,1] = 2.0*x[0]**2 + np.exp(np.cos(10.0*x[0])*np.cos(5.0*x[1])**2) + np.exp(np.sin(7.5*x[2]*x[2]))
    return y


if len(sys.argv)>1:
    inputfile=sys.argv[1]
    print("Looking for input file:", inputfile)
    inputs=(np.loadtxt(inputfile)).transpose()
    #print("inputs:\n", inputs)

    if len(sys.argv)>2:
        noise=float(sys.argv[2])
        print("noise set to:",noise)
    else:
        noise=0.0
        print("noise not set, setting to:",noise)


    y=sim(inputs)

    outputfile="toysim3D_output"
    print("outputs to output file:", outputfile)
    np.savetxt(outputfile, y, delimiter=' ', fmt='%1.4f')
else:
    print("Please provide input filename e.g.\npython toy-sim.py toy-sim_input\nand, optionally, noise amplitude e.g.\npython toy-sim.py toy-sim_input 0.25")
