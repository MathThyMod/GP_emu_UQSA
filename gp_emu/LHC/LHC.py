###############################
## optimised Latin hypercube ##
###############################
# 'config' file should specify:
# Number of input dimensions p 
# Number of inputs points desired n 
# Number of LHCs to be generated N 
# min and max of each dimension

import sys
import numpy as np

## look for LHC config file
if len(sys.argv)>1:
    configfile=sys.argv[1]
    print("config file:",configfile)
else:
    configfile='LHC_config.txt'
    print("config file not set, setting to:", configfile)

## read into a dictionary
print("Reading config file...")
config = {}
with open(configfile, 'r') as f:
    for line in f:
        (key, val) = line.split()
        config[key] = val
    print(config) 
## NEED CLOSING?

#print(config.keys())
dim = int(config['dim'])
n = int(config['n'])
N = int(config['N'])
filename = str(config['filename'])

minmax = []
for i in range(1,dim+1):
    templist = ( float(config['min'+str(i)]) , float(config['max'+str(i)]) )
    minmax.append(templist)
#print(minmax)
inputs = np.array(minmax)
print("Sim-input ranges:\n" , inputs)

#inputs_map = inputs ## this line will break later stuff because pass by ref
#for i in range(0,dim):
#    #inputs_map[:,i] = (inputs[:,i] -  inputs[0,i])/(inputs[1,i]-inputs[0,i])
#    inputs_map[i,:] = (inputs[i,:] -  inputs[i,0])/(inputs[i,1]-inputs[i,0])
#print(inputs)

print("Generating oLHC samples...")
# for each dimension i, generate n (no. of inputs) random numbers u_i1, u_i2
# as well as random purturbation of the integers b_i1 -> b_in : 0, 1, ... n-1
u=np.zeros((dim,n))
b=np.zeros((dim,n), dtype=np.int)
x=np.zeros((dim,n,N))
# produce the numbers x
for k in range(0,N):
    for i in range(0,dim):
        u[i,:] = np.random.uniform(0.0, 1.0, n)
        b[i,:] = np.arange(0,n,1)
        np.random.shuffle(b[i,:])
        x[i,:,k] = ( b[i,:] + u[i,:] ) / float(n)

print("Applying criterion...")
# do criterion test to find maximum of minimum distance
C=np.zeros(N)
C[:] = 10 # set high impossible original max distance
for k in range(0,N):
    for j1 in range(0,n):
        for j2 in range(0,n):
            val = np.linalg.norm( x[:, j1 , k] - x[: , j2, k] )
            if val < C[k] and j1 != j2:
                C[k] = val

K = np.argmax(C)
D = x[: , : , K]
#print("Optimal LHC is " , K, " with D:\n" , D)

print("Saving inputs to file...")
#np.savetxt('emu-input.txt', D, delimiter=" ", fmt='%1.4f')
# unscale the simulator input
for i in range(0,dim):
    D.T[:,i] = D.T[:,i]*(inputs[i,1]-inputs[i,0]) + inputs[i,0]
np.savetxt(filename, D.T, delimiter=" ", fmt='%1.4f')

print("DONE!")
