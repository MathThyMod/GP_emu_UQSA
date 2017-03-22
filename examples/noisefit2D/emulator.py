import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import gp_emu_uqsa.design_inputs as d
import gp_emu_uqsa.noise_fit as gn


#######################
## create noisy data ##
#######################

def mfunc(x): # function
    return 3.0*x[0]**3 + np.exp(np.cos(10.0*x[1])*np.cos(5.0*x[0])**2)

def nfunc(x): # noise function
    n = 0.500 * ( (x[1]**1)*(np.cos(6*x[0])**2 + 0.1) )
    return n
    
#### configuration of design inputs
dim, n, N = 2, 500, 200
minmax = [ [0.0,1.0] , [0.0,1.0] ]
filename = "INPUTS"
d.optLatinHyperCube(dim, n, N, minmax, filename)

#### we would perform our simulations here instead
x = np.loadtxt("INPUTS") # inputs
y = np.array([mfunc(xi) for xi in x]) # mean function
n = np.array([nfunc(xi) for xi in x]) # noise function
y = y + n*np.random.randn(y.size)
np.savetxt("OUTPUTS",y)


###################
## fit the noise ##
###################

data, noise = "config-data" , "config-noise"
gn.noisefit(data, noise, stopat=10, olhcmult=200)


###########
## plots ##
###########

# plotting
inputs = np.loadtxt("noise-inputs")
x , y = inputs[:,0] , inputs[:,1]
z = np.loadtxt("noise-outputs")[:,0]
size = z.size

# Set up plotting space
fig, ax = plt.subplots(nrows = 1, ncols = 2)

# Set up a regular grid of interpolation points
xi, yi = np.linspace(x.min(), x.max(), size), np.linspace(y.min(), y.max(), size)
xi, yi = np.meshgrid(xi, yi)

# Interpolate - known function
print("noise function")
zfun = nfunc(inputs.T)
fun = scipy.interpolate.Rbf(x, y, zfun, function='linear')
zf = fun(xi, yi)
zfp = ax[0].imshow(zf, vmin=zfun.min(), vmax=zfun.max(), origin='lower',
         extent=[x.min(), x.max(), y.min(), y.max()])
ax[0].scatter(x, y, c=zfun)
ax[0].set_title("noise function")
plt.colorbar(zfp, ax=ax[0])

# Interpolate - noise fit
print("noise fit")
fit = scipy.interpolate.Rbf(x, y, z, function='linear')
zn = fit(xi, yi)
if False: ## color with fit
    znp = ax[1].imshow(zn, vmin=z.min(), vmax=z.max(), origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
    ax[1].scatter(x, y, c=z)
else: ## color with func
    print("points are coloured by *true* noise value, not the fit")
    print("colorbar set to range of *true* noise value, not the fit")
    znp = ax[1].imshow(zn, vmin=zfun.min(), vmax=zfun.max(), origin='lower',
             extent=[x.min(), x.max(), y.min(), y.max()])
    ax[1].scatter(x, y, c=zfun)
ax[1].set_title("noise fit")
plt.colorbar(znp, ax=ax[1])

plt.show()

