# GP_emu
________

GP_emu is designed to build, train, and validate a Gaussian Process Emulator via a series of simple routines:

1. The emulator is built from a user specified configuration file and beliefs file, and choice of kernel (covariance function)

2. The emulator is trained and validated on subsets of data

3. A full prediction (posterior distribution) is made in the input data range

The subpackage GP_emu.design_inputs also contains routines for designing input data for simulations, the results of which are intended for building an emulator.

In the future, GP_emu will also include sensitivity analysis using GP emulators.

GP_emu is written in Python, and should function in both Python 2.7+ and Python 3.


Table of Contents
=================
* [Installation](#Installation)
* [Building an Emulator](#Building an Emulator)
  * [Example](#Example)
  * [Main Script](#Main Script)
  * [Config File](#Config File)
  * [Beliefs File](#Beliefs File)
* [Design Input Data](#Design Input Data)

<a name="Installation"/>
## Installation
Install with
```
python setup.py install
```

The following additional packages will be installed:
*numpy, scipy, matplotlib, future*

<a name="Building an Emulator"/>
## Building an Emulator
The user must write a configuration file, a beliefs file, and a Python script.

<a name="Example"/>
### Example
To run an example, do
```
cd examples/toy-sim/
python emulator.py
```
<a name="Main Script"/>
### Main Script
This script runs a series of functions in GP_emu which automatically perform the main tasks outlined above. This allows flexibility for the user to create several different scripts for trying to fit an emulator to their data.

```
import gp_emu as g

#### configuration file
conf = g.config("toy-sim_config")

#### setup emulator
emul = g.setup(conf, K)

#### train emulator and run validation diagnostics
g.training_loop(emul, conf)

#### build final version of emulator
g.final_build(emul, conf)

#### plot full prediction, "mean" or "var"
g.plot(emul, [0,1],[2],[0.65], "mean")
```

The configuration file is explained later.

#### training and validation
The functions ```training_loop()``` and ```final_build()``` are very similar, but differ:

* ```training_loop()``` trains the emulator on the current training data and validates against the current set of validation data. Prompts for retraining on new validation sets (after including the last validation set in the training set) will be made until there are no more validation sets left.

* ```final_build()``` will include the most recently used validation data set in the training set, and rebuild the emulator with this new larger training data set. No diagnostics are performed.

*  An additional 3rd argument ```auto``` can be added to both ```training_loop()``` and ```final_build()``` to toggle (or explicitly set) whether the subsequent training runs and the final build will proceed automatically e.g. ```g.training_loop(emul, conf, True)``` or ```g.training_loop(emul, conf, auto=True)``` or ```g.training_loop(emul, conf, auto=False)``` etc. The default value for auto, if it is absent, is True.


#### Plotting
The full prediction (posterior distribution), either the mean or the variance, is displayed as a plot. Plots can be 1D (scatter plot) or 2D (colour map). For a 2D plot:

* the first list is input dimensions for (x,y) of the plot

* the second list is input dimensions to set constant values

* the third list is these constant values

<a name="Config File"/>
### Config File
The configuration file does two things:

1. Specifies the beliefs file and data files

  * the beliefs file is explained in detail below

  * the inputs file is rows of whitespaces-separated input points, each column in the row corresponding to a different input dimensions

  * the output file is rows of output points; only one dimensional output may be specified, so each row should be a single value

2. Specifies how the emulator is trained on the data; see below

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
Specifies how the data is divided up into training and validation sets. The data is randomly shuffled before being divided into sets (an option to turn this off may be introduced later for the purposes of training on time-series).

1. first value -- __10__ 0 2 -- how many sets to divide data into (determines size of validation set)

2. second value -- 10 __0__ 2 -- which V-set to initially validate against (currently, this should usually be set to zero; this mostly redundant feature is here for future implementation of fitting time-series data).

3. third value -- 10 0 __2__ -- number of validation sets (determines number of training points)

| tv_config | data points | T points | V-set size | V sets  |
| ----------| ------------| -------- | ---------- | ------- |
| 10 0 2    | 200         | 160      | 20         | 2       |
| 4 0 1     | 100         | 75       | 25         | 1       |
| 10 0 1    | 100         | 90       | 10         | 1       |

#### delta_bounds and sigma_bounds
Sets bounds on the hyperparameters while fitting the emulator. These bounds will only be used if constraints are specified True i.e. if
```
constraints T
```

Leaving delta_bounds and sigma_bounds 'empty'
```
delta_bounds [ ]
sigma_bounds [ ]
```
automatically constructs reasonable bounds on delta and sigma, though these might not be appropriate for the problem at hand (in which case, set constraints to false)
```
constraints F
```

To explicitly set bounds, a list of lists must be constructed, the inner lists specifying the lower and upper range on each hyperparameter, with the inner lists in the order that the hyperparameters are effectively defined due to the kernel definition.

##### delta bounds

| input dimension | kernel | delta_bounds |
| --------------- | ------ | ------------ |
| 3 | __gaussian__ + gaussian | [ __[0.0,1.0] , [0.1,0.9], [0.2,0.8]__, [0.0,1.0] , [0.1,0.9], [0.2,0.8] ] |
| 3  | gaussian + noise | [ [0.0,1.0] , [0.1,0.9], [0.2,0.8] ]  |
| 2  | 2_delta_per_dim + __gaussian__ | [ [0.0,1.0] , [0.1,0.9], _[0.0,1.0] , [0.1,0.9]_ , __[0.0,1.0] , [0.1,0.9]__ ] |

For 2_delta_per_dim there are two delta for each input dimension, so the list requires the first delta for each dimension to be specified first, followed by the second delta for each dimension i.e.
```
[ [d1(0) range] , [d1(1) range] , [d2(0) range] , [d2(1) range] ]
```

##### sigma bounds

Sigma_bounds works in the same way as delta_bounds, but is simpler since there is one sigma per kernel:

| input dimension | kernel     | sigma_bounds |
| --------------- | ---------- | ------------ |
| 2  | __gaussian__ + gaussian | [ [10.0,70.0] , [10.0,70.0] ] |
| 3  | gaussian + noise        | [ [10.0,70.0] , [0.001,0.25] ] |
| 1  | arbitrary_kernel        | [ [10.0,70.0] ] |


#### fitting options

* __tries__ : is how many times (interger) to try to fit the emulator for each training run
e.g. ``` tries 5 ```

* __constraints__ : is whether to use constraints: must be either true ```constraints T``` or false ```constraints F```

* __stochastic__ : is whether to use a stochastic 'global' optimiser ```stochastic T``` or a gradient optimser ```stochastic F```. The stohcastic optimser is slower but for well defined fits usually allows fewer tries, whereas the gradient optimser is faster but requires more tries to assure the optimum fit is found

* __constraints_type__ : can be ```constraints_type bounds``` (use the specified delta_bounds and sigma_bounds), ```constraints_type noise``` (fix the noise; only works if the last kernel is noise), or ```constraints_type standard``` (standard constraints are set to keep delta above a very small value, for numerical stability - any option that isn't bounds or noise will set constraints to standard).

<a name="Beliefs File"/>
### Beliefs File
The beliefs file specifies beliefs about the data, namely which input dimensions are active, what the mean function is believed to be, and the initial beliefs about the hyperparameter values.
```
active all
basis_str 1.0 x
basis_inf NA 0
beta 1.0 1.0
fix_mean F
kernel gaussian() noise()
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
beta 1.0 1.0 1.0
```

For mean function m(__x__) = b0 + b0x0 + b1x1^2 + b2x2^3
```
basis_str 1.0 x   x**2 x**3
basis_inf NA  0   1    2
beta      1.0 2.0 1.1  1.6
```
In this last example, spaces have been inserted for readability.

Bear in mind that the initial values of beta, while needing to be set, do not affect the emulator fitting. However, for consistency with the belief files produced after fitting the data, which may be used to reconstruct the emulator for other purposes or may simply be used to store the fit parameters, the beta hyperparameters must be set in the initial belief file. They can all be set to 1.0, for simplicity.

The __fix_mean__ option simply allows for the mean to remain fixed at its initial specifications in the belief file. It must be ```fix_mean T``` or ```fix_mean F```.


#### Kernels
The currently available kernels are

| kernel   | class      | description |
| -------- | -----------| ----------- |
| gaussian | gaussian() | gaussian kernel |
| noise    | noise()    | additive uncorrelated noise |
| test     | test()     | (useless) example showing two length scale hyperparameters (delta) per input dimension |

More kernels can be built easily, and the following will be available in the near future: Matern, Rational Quadratic.

Kernels can be added togeter to create new kernels e.g. ```gaussian() + noise()```, which is implemented via operator overloading. To specify a list of kernels to be added together, list them in the beliefs file, separated by whitespace:

```
kernel gaussian() noise()
```

Other kernel combination operations e.g. multiplication could also be implemented, but is not priority.


#### the kernel hyperparameters
The kernel hyperparameters will be automatically constructed if the lists are left empty i.e.
```
delta [ ]
sigma [ ]
```
which is recommended as the initial values do not affect how the emulator is fit. However, for consistency with the beliefs file produced after training (and to explain that file), the kernel hyperparameter beliefs can be specified below. However, the easiest way to construct these lists is to run the program with empty lists, and then copy and paste the lists from the updated beliefs files into the initial belief file (specified in the configuration file).

##### delta
The following shows how to construct the lists piece by piece.

1. a list for each kernel being used e.g. K = gaussian() + noise() we need ```delta [ [ ] , [ ] ]```

2. within each kernel list, d lists, where d is the number of hyperparameters per dimension
 * if there is one delta per input dimension for K = one_delta_per_dim() we need ```[ [ [ ] ] ]```
 * if there are two delta per input dimenstion for K = two_delta_per_dim() we need  ```[ [ [ ] , [ ] ] ]``` i.e. within the kernel list we have two lists in which to specify the delta for the first input dimension and the second input dimension
 * so for K = two_delta_per_dim() + one_delta_per_dim() we need ```[  [ [ ],[ ] ]  ,  [ [ ] ]  ]```

Within these inner most lists, the n values of delta (n is the number of dimensions) should be specified.

e.g. K = one_delta_per_dim() in 1D we need ```[ [ [1.0] ] ]```

e.g. K = one_delta_per_dim() in 2D we need ```[ [ [1.0, 1.0] ] ]```

e.g. K = two_delta_per_dim() in 1D we need  ```[ [ [1.0] , [1.0] ] ]```

e.g. K = two_delta_per_dim() in 2D we need  ```[ [ [1.0,1.0] , [1.0,1.0] ] ]```

e.g. K = gaussian() + gaussian() in 1D we need ```[ [ [1.0] ] , [ [1.0] ] ]```

e.g. K = gaussian() + gaussian() in 2D we need ```[ [ [1.0,1.0] ] , [ [1.0, 1.0] ] ]```

e.g. K = gaussian() + noise() in 2D we need
```
delta [ [ [0.2506, 0.1792] ] , [ ] ]
```
_If a kernel has no delta values, such as the noise kernel, then its list should be left empty._

##### sigma
Sigma is simpler, as there is one per kernel:

e.g. K = gaussian() in 1D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() in 2D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() + noise() in 1D we need ``` sigma [ [0.6344] , [0.0010] ]```

e.g. K = gaussian() + noise() in 2D we need ``` sigma [ [0.6344] , [0.0010] ]```

<a name="Design Input Data"/>
## Design Input Data
To import this subpackage use something like this
```
import gp_emu.design_inputs as d
```
Currently, only an optimised Latin Hypercube design is included.

Scripts should be written to configure the design and run the design function e.g.

```
import gp_emu.design_inputs as d

#### configuration of design inputs
dim = 2
n = 60
N = 200
minmax = [ [0.0,1.0] , [0.0,1.0] ]
filename = "toy-sim_input"

#### call function to generate input file
d.optLatinHyperCube(dim, n, N, minmax, filename)
```
The design input points, output to _filename_, are suitable for reading by GP_emu: each line (row) represents one input point of _dim_ dimensions.
