# GP_emu

## Install
Install with
```
python setup.py install
```

The following additional packages will be installed:
*numpy, scipy, matplotlib, future*

## Example
To run an example, do
```
cd examples/toy-sim/
python emulator.py
```

## Overview
GP_emu is designed to build, train, and validation a Gaussian Process Emulator via a series of simple routines:  
1. The emulator is built from a user specified configuration file and choice of kernel (covariance function)

2. The emulator is trained and validated on a subsets of data

3. A full prediction (posterior distribution) is made in the input data range

The user must write a configuration file, a beliefs file, and a Python script.

### Python Script
This script runs a series of functions in GP_emu which automatically perform the main tasks outlined above. This allows flexibility for the user to create several different scripts for trying to fit an emulator to their data.

```
import gp_emu as g

#### configuration file
conf = g.config_file("toy-sim_config")

#### define kernel
K = g.gaussian() + g.noise()

#### setup emulator
emul = g.setup(conf, K)

#### train emulator and run validation diagnositcs
g.training_loop(emul, conf)

#### build final version of emulator
g.final_build(emul, conf)

#### plot full prediction, "mean" or "var"
g.plot(emul, [0,1],[2],[0.65], "mean")
```
#### Kernels
The available kernels can be added togeter to create new kernels, as shown above. Kernels cannot currently be multiplied together, but this can be easily implemented.

#### Plotting
Currently the full prediction is displayed as a plot, although the full posterior is not saved to file. Either the posterior mean or the posterior variance can be plotted. Plots can be 1D (scatter plot) or 2D (colour map).  
For the 2D case, the first list specifies the data input dimensions for the (x,y) of the plot, the second list specifies which input dimensions will be held at a constant value, and the last list specifies these constant values.


### Config File
The configuration file does two things:
1. Specifies the beliefs file and data files
2. Allows a lot of control over how the emulator is trained

```
beliefs toy-sim_beliefs
inputs toy-sim_input
outputs toy-sim_output
tv_config 10 0 2
delta_bounds [ ]
sigma_bounds [ ]
tries 1
constraints T
stochastic T
constraints_type bounds
```
#### tv_config
The Training-Validation configuration specifies how the data is divided up into training and validation sets. Currently, the data is randomly shuffled before being divided into sets, though an option to turn this off may be introduced later for the purposes of training on time-series.  
1. The first value e.g. __10__ 0 2 is how many sets the data is to be divided into.
2. The second value  e.g. 10 __0__ 2 is which validation set to initially train against (currently, this should be set to zero; this option is currently mostly redundant, but is included for the purposes of training on time-series data).
3. The third value  e.g. 10 0 __2__ is how many sets are required.

e.g. 200 data points and tv_config 10 0 2 would give 160 training points and 2 sets of 20 validation points, and the first validation set would be used during the first round of training for the validation diagnositcs

#### hyperparameter bounds
Leaving delta_bounds and sigma_bounds 'empty' i.e. [] automatically constructs bounds on delta and sigma to be used for fitting the emulator. However, these bounds will only be used if constraints are specified i.e. constraints T, see below. To explicitly set bounds, within the list there must be lists specifying the lower and upper range on each hyperparameter, with the hyperparameters listed in the order that the kernel is defined, such that we have an ordered list of all of out deltas bounds in the order that delta are effectively specified.

For 3 dimensional input with Kernel = gaussian() + noise() we need delta_bounds [ [0.0,1.0] , [0.1,0.9], [0.2,0.8] ] since there is a single delta per input dimension and noise has no delta values. We would set sigma_bounds [ [10.0,70.0] , [0.001,0.25] ] since there is one sigma for the Gaussian kernel and one sigma for the delta kernel.

For 3 dimensional input with Kernel = gaussian() + gaussian() we need  [ __[0.0,1.0] , [0.1,0.9], [0.2,0.8]__, *[0.0,1.0] , [0.1,0.9], [0.2,0.8]* ] where the bounds for delta from the first and second kernel are shown in bold and italics respectively.

For 2 dimensional input with Kernel = two_delta_per_dim() i.e. there are two delta for each input dimension for that single kernel, we need [ __[0.0,1.0] , [0.1,0.9]__, *[0.0,1.0] , [0.1,0.9]* ] where the bounds for the first delta of the kernel and second dimension of the kernel are shown in bold and italics respectively.

#### fitting options
* __tries__ is how many times (interger) to try to fit the emulator for each training run
* __constraints__ is whether to use constraints: must be either *T* (true) or *F* (false)
* __stochastic__ is whether to use a stochastic 'global' optimiser (set *T*) or a gradient optimser (set *F*). The stohcastic optimser is slower but for well defined fits usually allwos fewer tries, whereas the gradient optimser is faster but requires more tries to assure the optimum fit is found
* __constraints_type__ can be _bounds_ (use the specified bounds), _noise_ (fix the noise; only works if the last kernel is noise), or _none_ (standard constraints are set to keep delta above a very small value, for numerical stability)

### Beliefs File
The beliefs file specifies beliefs about the data, namely which input dimensions are active, what the mean function is believed to be, and the initial beliefs about the hyperparameter values.
```
active all
basis_str 1.0 x
basis_inf NA 0
beta 1.0 1.0
fix_mean F
delta [ ]
sigma [ ]
```
#### the mean function
This must be specified via __basis_str__ and __basis_inf__ which together define the form of the mean function. __basis_str__ defines the functions making up the mean function, and __basis_inf__ defines which input dimension those functions are for. __beta__ defines the values of the mean function hyperparameters

For mean function m(__x__) = b0
```
basis_str 1.0
basis_inf NA
beta 1.0
```

For mean function m(__x__) = 0
```
basis_str 0.0
basis_inf NA
beta 1.0
```

For mean function m(__x__) = b0 + b0x0 + b2x2
```
basis_str 1.0 x x
basis_inf NA 0 2
beta 1.0
```

For mean function m(__x__) = b0 + b0x0 + b1x1^2 + b2x2^3
```
basis_str 1.0 x   x**2 x**3
basis_inf NA  0   1    2
beta      1.0 2.0 1.1  1.6
```
In this last example, spaces have been inserted for readability.

Bear in mind that the initial values of beta, while needing to be set, do not affect the emulator fitting. However, for consistency with the belief files produced after fitting the data, which may be used to reconstruct the emulator for other purposes or may simply be used to store the fit parameters, the beta hyperparameters must be set in the initial belief file. They can all be set to 1.0, for simplicity.

The __fix_mean__ option simply allows for the mean to remain fixed at its initial specifications in the belief file. In this case, the beta hyperparameters must be chosen carefully.

#### the kernel hyperparameters
The kernel hyperparameters will be automatically constructted if the lists are left empty i.e. [] which is recommended as the initial values do not affect how the emulator is fit. However, for consistency with the beliefs file produced after training (and to explain that file), the kernel hyperparameter beliefs can be specified as:

1. a list for each kernel being used e.g. for K = gaussian() + noise() we need [ __[]__ , __[]__ ]
2. within each kernel list, n*d lists of hyperparameters where n is the number of active input dimensions and d is the number of hyperparameters per dimension e.g.
 * if there is one delta per input dimension for K = one_delta_per_dim() we need [ [ __[]__ ] ]
 * if there are two delta per input dimenstion for K = two_delta_per_dim() we need  [ [ __[]__ , __[]__ ] ] i.e. within the kernel list we have two lists in which to specify the delta for the first input dimension and the second input dimension
 * so for K = two_delta_per_dim() + one_delta_per_dim() we need [ [ [], [] ], [] ]
3. if a kernel has no delta values, such as the noise kernel, then its list should be left empty
