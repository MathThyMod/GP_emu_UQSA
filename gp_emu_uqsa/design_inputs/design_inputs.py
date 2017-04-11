###############################
## optimised Latin hypercube ##
###############################
# Number of input dimensions p 
# Number of inputs points desired n 
# Number of LHCs to be generated N 
# min and max of each dimension
# filename for results

import numpy as _np
import scipy.spatial.distance as _dist

def optLatinHyperCube(dim=None, n=None, N=None, minmax=None, filename="inputs", fextra=None):
    """Design input data using an optimisated latin hypercube design and save it to a file.

    Args:
        dim (int): Dimensions of data points
        n (int): Number of data points
        N (int): Number of designs to create. The best design is chosen.
        minmax (list): Value interval on each dimension e.g.  [ [0.0,1.0] , [0.0,1.0] ]
        filename (str): Name of file
        fextra (nparray): Numpy array of data with same number of columns as dim

    Returns:
        None

    """

    ## check the arguments to the function
    print('dim:' , dim)
    print('n:' , n)
    print('N:' , N)
    print('minmax:' , minmax)
    print('filename:' , filename)

    if dim==None or n==None or N==None or minmax==None:
        print("Please supply values for function arguments (default for filename is \"inputs\")")

    if len(minmax) != dim:
        print("WARNING: length of 'minmax' (list of lists) must equal 'dim'")
        exit()

    if fextra is not None:
        #try:
        xex = fextra
        #except FileNotFoundError as e:
        #    print("ERROR: file", fextra, "for inputs and/or outputs not found. Exiting.")
        #    exit()
        print("\nGenerating", N, "oLHC samples of", n ,"points, combining with supplied extra data, and checking maximin criterion (pick design with maximum minimum distance between design points)...")
    else:
        print("\nGenerating", N, "oLHC samples of", n ,"points and checking maximin criterion (pick design with maximum minimum distance between design points)...")
    # for each dimension i, generate n (no. of inputs) random numbers u_i1, u_i2
    # as well as random purturbation of the integers b_i1 -> b_in : 0, 1, ... n-1
    u=_np.zeros((n,dim))
    b=_np.zeros((n,dim), dtype=_np.int)
    x=_np.zeros((n,dim))

    # produce the numbers x
    for k in range(0,N):
        for i in range(0,dim):
            u[:,i] = _np.random.uniform(0.0, 1.0, n)
            b[:,i] = _np.arange(0,n,1)
            _np.random.shuffle(b[:,i])
            x[:,i] = ( b[:,i] + u[:,i] ) / float(n)

        # add extra data if present
        if fextra is not None:
            xt = _np.concatenate([x,xex])
        else:
            xt = x

        # calculate and check maximin
        maximin = _np.argmin( _dist.pdist(xt,'sqeuclidean') )
        if k==0 or maximin > best_maximin:
            best_D = _np.copy(x)
            best_k = k
            best_maximin = maximin
                
    D = best_D
    print("Optimal LHC design was no." , best_k)#, " with D:\n" , D)

    print("Saving inputs to file...")
    # unscale the simulator input
    inputs = _np.array(minmax)
    for i in range(0,dim):
        D[:,i] = D[:,i]*(inputs[i,1]-inputs[i,0]) + inputs[i,0]
    # save to file
    _np.savetxt(filename, D, delimiter=" ", fmt='%.8f')

    print("DONE!")
    return None
