# GP_emu

```
============    _(o> ====================
   GP_emu    ¬(GP)      by Sam Coveney   
============   ||    ====================
```
________

GP_emu is designed for building, training, and validating a Gaussian Process Emulator via a series of simple routines. It is supposed to encapsulate the [MUCM methodology](http://mucm.aston.ac.uk/toolkit/index.php?page=MetaHomePage.html), while also allowing the flexibility to build general emulators from combinations of kernels. In special cases, the trained emulators can be used for uncertainty and sensitivity analysis.

GP_emu is written in Python, and should function in both Python 2.7+ and Python 3. To install GP_emu, download the package and run the following command inside the top directory:
```
python setup.py install
```

Table of Contents
=================
* [Building an Emulator](#Building an Emulator)
  * [Main Script](#Main Script)
  * [Config File](#Config File)
  * [Beliefs File](#Beliefs File)
  * [Create files automatically](#Create files automatically)
  * [Fitting the emulator](#Fitting the emulator)
  * [Reconstruct an emulator](#Reconstruct an emulator)
* [Design Input Data](#Design Input Data)
* [Uncertainty and Sensitivity Analysis](#Uncertainty and Sensitivity Analysis)
* [Examples](#Examples)
  * [Simple toy simulator](#Simple toy simulator)
  * [Sensitivity examples](#Sensitivity examples)

<a name="Building an Emulator"/>
## Building an Emulator
GP_emu uses the [methodology of MUCM](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadCoreGP.html) for building, training, annd validating an emulator.

The user should create a project directory (separate from the GP_emu download), and within it place a configuration file, a beliefs file, and an emulator script containing GP_emu routines (the directory and these files can be created automatically - see [Create files automatically](#Create files automatically)). Separate inputs and outputs files should also be placed in this new directory.

The idea is to specify beliefs (e.g. form of mean function, values of hyperparameters) in the beliefs file, specify filenames and fitting options in the configuration file, and call GP_emu routines with a script. The main script can be easily editted to specifiy using different configuration files and beliefs files, allowing the user's focus to be entirely on fitting the emulator.

NOTE: emulators should be trained using a design data set such as an optimized latin hypercude design. See the section [Design Input Data](#Design Input Data).

<a name="Main Script"/>
### Main Script
This script runs a series of functions in GP_emu which automatically perform the main tasks outlined above. This allows flexibility for the user to create several different scripts for trying to fit an emulator to their data.

```
import gp_emu as g

#### setup emulator with configuration file
emul = g.setup("toy-sim_config")

#### train emulator and run validation diagnostics
g.train(emul)

#### plot full prediction, "mean" or "var"
g.plot(emul, [0,1], [2], [0.65], "mean")
```

The configuration file is explained later.

#### setup
The setup function needs to be supplied with a configuation file, and can optionally be supplied with two extra options e.g.:
```
emul = g.setup(configfilename, datashuffle = True, scaleinputs = True)
```

The option ```datashuffle = False``` (default ```True```) prevents the random shuffling of the inputs and outputs (which may be useful when the data is a timeseries or when the user wishes to work on the same subset of data).

The option ```scaleinputs = False``` (default ```True```) prevents each input dimension from being scaled into the range 0 to 1(or scaled by the optional ```input_minmax``` in the beliefs file).


#### train
The function ```train()``` will perform training and validation diagnostics.

* if there are validation data sets available, then the emulator trains on the current training data and validates against the current validation set. Afterwards, a prompt will ask whether the user wants to include the current validation set into the training set (recommended if validation diagnostics were bad) and retrain.

* An optional argument ```auto``` (default ```True```) can be given to toggle on/off the automatic retraining of the emulator with the validation data included in the training data.

* An optional argument ```message``` (default ```False```) can be given to toggle on/off the printing of messages from underlying optimization routines (these messages may help identify problems with fitting an emulator to the data).


#### plot
The full prediction (posterior distribution), either the mean or the variance, can be displayed as a plot. Plots can be 1D line plots or 2D colour maps.

* 2D plots: the first argument is a list of which input dimensions to use for the (x,y)-axes

* 1D plots: the first argument is a list of the input dimension to use for the x-axis

* The second argument is a list of which input dimensions to set to constant values, and the third argument is a list of these constant values.

* The third optional argument (default ```"mean"```) specifies whether to plot the mean (```"mean"```) or the variance (```"var"```).

* The fourth optional argument is a list of strings for the axes labels.

e.g. for a 1D plot of the variance for input 0 varying from 0 to 1 and inputs 1 and 2 held fixed at 0.10 and 0.20 respectively:
```
g.plot(emul, [0], [1,2], [0.10,0.20], "var", ["input 0"])
```

<a name="Config File"/>
### Config File
The configuration file does two things:

1. Specifies the name of the beliefs file and data files

  * the beliefs file is specified in the [Beliefs File](#Beliefs File) section

  * inputs (outputs) file: each row corresponds to a single data point. Whitespace separates each dimension of the input (output). Naturally, the i'th row in the inputs file corresponds to the i'th row in the outputs file.

  ```
  beliefs toy-sim_beliefs
  inputs toy-sim_input
  outputs toy-sim_output
  ```

2. Specifies options for training the emulator on the data e.g.
  ```
  tv_config 10 0 2
  delta_bounds [ ]
  sigma_bounds [ ]
  tries 5
  constraints T
  stochastic F
  constraints_type bounds
  ```

*The configuration file can be commented*, making it easy to test configurations without making a whole new config file e.g.
```
#delta_bounds [ [0.005 , 0.015] ]
#sigma_bounds [ [0.100 , 3.000 ] ]
delta_bounds [ [0.001 , 0.025] ]
sigma_bounds [ [0.010 , 4.000 ] ]
```


#### tv_config
Specifies how the data is split into training and validation sets.

1. first value -- __10__ 0 2 -- how many sets to divide the data into (determines size of validation set)

2. second value -- 10 __0__ 2 -- which V-set to initially validate against (currently, this should be set to zero; this mostly redundant feature is here to provide flexibility for implementing new features in the future)

3. third value -- 10 0 __2__ -- number of validation sets (determines number of training points)

| tv_config | data points | T points | V-set size | V sets  |
| ----------| ------------| -------- | ---------- | ------- |
| 10 0 2    | 200         | 160      | 20         | 2       |
| 4 0 1     | 100         | 75       | 25         | 1       |
| 10 0 1    | 100         | 90       | 10         | 1       |


#### delta_bounds and sigma_bounds
Sets bounds on the hyperparameters while fitting the emulator. These bounds will only be used if ```constraints T``` and ```constraints_type bounds```, but *the constraints are used as intervals in which to generate initial guesses for fitting*.

Leaving delta_bounds and sigma_bounds empty, i.e. ```delta_bounds [ ]``` and ```sigma_bounds [ ]```, automatically constructs bounds on delta and sigma, though these might not be suitable.

To explicitly set bounds, a list of lists must be constructed, the inner lists specifying the lower and upper range on each hyperparameter, with the inner lists in the order that the hyperparameters are defined by the kernel definition.


##### delta bounds
| input dimension | kernel | delta_bounds |
| --------------- | ------ | ------------ |
| 3  | __gaussian__ + gaussian | [ __[0.0,1.0] , [0.1,0.9], [0.2,0.8]__, [0.0,1.0] , [0.1,0.9], [0.2,0.8] ] |
| 3  | gaussian + noise | [ [0.0,1.0] , [0.1,0.9], [0.2,0.8] ]  |
| 2  | 2_delta_per_dim + __gaussian__ | [ [0.0,1.0] , [0.1,0.9], _[0.0,1.0] , [0.1,0.9]_ , __[0.0,1.0] , [0.1,0.9]__ ] |

For a kernel with two delta for each input dimension e.g. 2_delta_per_dim, so the list requires the first delta for each dimension to be specified first, followed by the second delta for each dimension e.g.
```
[ [d1(0) range] , [d1(1) range] , [d2(0) range] , [d2(1) range] ]
```

##### sigma bounds
| input dimension | kernel     | sigma_bounds |
| --------------- | ---------- | ------------ |
| 2  | __gaussian__ + gaussian | [ __[10.0,70.0]__ , [10.0,70.0] ] |
| 3  | gaussian + __noise__        | [ [10.0,70.0] , __[0.001,0.25]__ ] |
| 1  | __kernel_with_two_sigma__ + noise   | [ __[10.0,70.0] , [11.0, 71.1]__ , [0.001,0.25] ] |

So ```sigma_bounds``` works in the same way as ```delta_bounds```, but is simpler because the number of sigma don't depend on the number of input dimensions:




#### fitting options

* __tries__ : is how many times (integer) to try to fit the emulator for each round of training e.g. ``` tries 5 ```

* __constraints__ : is whether to use constraints during fitting: either true ```constraints T``` or false ```constraints F```

* __stochastic__ : is whether to use a stochastic 'global' optimiser ```stochastic T``` or a gradient optimser ```stochastic F```. The stochastic optimser is slower but usually allows fewer tries, the gradient optimser is faster but requires more tries. *The stochastic optimiser always constrains the search using the hyperparameter bounds.*

* __constraints_type__ : can be ```constraints_type bounds``` (use the specified delta_bounds and sigma_bounds), ```constraints_type noise``` (fix the noise; only works if the last kernel is ```noise()```), or the default option ```constraints_type standard``` (standard constraints are set to keep delta above a very small value, for numerical stability).


<a name="Beliefs File"/>
### Beliefs File
The beliefs file specifies beliefs about the data, namely which input dimensions are active, what mean function to use, and values of the hyperparameters (before training - this doesn't affect the training).
```
active all
output 0
basis_str 1.0 x
basis_inf NA 0
beta 1.0 1.0
fix_mean F
kernel gaussian() noise()
delta [ ]
sigma [ ]
```

#### choosing inputs and outputs
The input dimensions to be used for the emulator are specified by ```active```. For all input dimensions use ```active all```, else list the input dimensions (indexing starts from 0) e.g. ```active 0 2```.

The output dimension to build the emulator for is specified by ```output```. Only a single index should be given e.g. ```output 2``` will use column '2' (technically the third column) of the output file.


#### the mean function
The specifications of ```basis_str``` and ```basis_inf``` define the mean function. ```basis_str``` defines functions and ```basis_inf``` defines which input dimension correspond to those functions. ```__beta__``` defines the mean function hyperparameters. The initial values of ```beta``` do not affect the emulator training, so they can be set to 1.0 for simplicity.

For mean function m(__x__) = b0
```
basis_str 1.0
basis_inf NA
beta 1.0
```

For mean function m(__x__) = 0
```
basis_str 1.0
basis_inf NA
beta 0.0
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

The mean function can be fixed at its initial specification using ```fix_mean T``` (useful to specify a zero mean), else to adjust the mean function during training use ```fix_mean F```


#### Kernels
The currently available kernels (defined in \_emulatorkernels.py) are

| kernel   | class      | description |
| -------- | -----------| ----------- |
| gaussian | gaussian() | gaussian kernel |
| gaussian_mucm | gaussian() | gaussian kernel - [this loglikelihood](http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=MetaFirstExamplePartC.html) will be used |
| noise    | noise()    | additive uncorrelated noise |
| test     | test()     | (useless) example showing two length scale hyperparameters (delta) per input dimension |


Kernels can be added togeter to create new kernels e.g. ```gaussian() + noise()```, which is implemented via operator overloading. To specify a list of kernels to be added together, list them in the beliefs file, separated by whitespace:

```
kernel gaussian() noise()
```

Nuggets can be specified e.g. for nugget = 0.001 use ```gaussian(0.001)``` (note there are _no whitespaces_ within the brackets).


#### the kernel hyperparameters
The kernel hyperparameters will be automatically constructed if the lists are left empty i.e.
```
delta [ ]
sigma [ ]
```
The initial values do not affect how the emulator is fit, but in order to understand the updated beliefs files they must be explained (the easiest way to construct these lists is to use empty list and then copy the new lists from the updated beliefs files).

##### delta
The following shows how to construct the lists piece by piece.

1. a list for each kernel being used e.g. K = gaussian() + noise() we need ```delta [ [ ] , [ ] ]```

2. within each kernel list, d lists, where d is the number of hyperparameters per dimension
 * if there is one delta per input dimension for K = one_delta_per_dim() we need ```[ [ [ ] ] ]```
 * if there are two delta per input dimenstion for K = two_delta_per_dim() we need  ```[ [ [ ] , [ ] ] ]``` i.e. within the kernel list we have two lists, to specify the delta for the first input dimension and to specify the delta for the second input dimension
 * so for K = two_delta_per_dim() + one_delta_per_dim() we need ```[  [ [ ],[ ] ]  ,  [ [ ] ]  ]```

Within these inner most lists, the n values of delta (n is the number of dimensions) should be specified.

e.g. K = one_delta_per_dim() in 1D we need ```[ [ [1.0] ] ]```

e.g. K = one_delta_per_dim() in 2D we need ```[ [ [1.0, 1.0] ] ]```

e.g. K = two_delta_per_dim() in 1D we need  ```[ [ [1.0] , [1.0] ] ]```

e.g. K = two_delta_per_dim() in 2D we need  ```[ [ [1.0,1.0] , [1.0,1.0] ] ]```

e.g. K = gaussian() + gaussian() in 1D we need ```[ [ [1.0] ] , [ [1.0] ] ]```

e.g. K = gaussian() + gaussian() in 2D we need ```[ [ [1.0,1.0] ] , [ [1.0, 1.0] ] ]```

e.g. K = gaussian() + noise() in 2D we need ``` delta [ [ [0.2506, 0.1792] ] , [ ] ] ```

_If a kernel has no delta values, such as the noise kernel, then its list should be left empty._

##### sigma
Sigma is simpler, as there are a fixed number per kernel (number of sigma doesn't increase with number of input dimensions). Again, the sigma must appear in the order that the kernel is specified:

e.g. K = gaussian() in 1D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() in 2D we need ``` sigma [ [0.6344] ]```

e.g. K = gaussian() + noise() in 5D we need ``` sigma [ [0.6344] , [0.0010] ]```

e.g. K = kernel_with_two_sigma() + noise() in 5D we need ``` sigma [ [0.6344 , 0.4436] , [0.0010] ]```


<a name="Create files automatically"/>
### Create files automatically
A routine ```create_emulator_files()``` is provided to create a directory containing default belief, config, and main script files. This is to allow the user to easily set up different emulators.

It is simplest to run this function from an interactive python session as follows:
```
>>> import gp_emu as g
>>> g.create_emulator_files()
```
The function will then prompt the user for input.


<a name="Fitting the emulator"/>
### Fitting the emulator
GP_emu uses Scipy and Numpy routines for fitting the hyperparameters. The file \_emulatoroptimise.py contains the routines *differential\_evolution* and *minimize*, which can take additional arguments which GP_emu (for simplicity) does not allow the user to specify at the moment. However, these additional arguments may make it easier to find the minimum of the negative loglikelihood function, and can easily be looked-up online and added to the code by the user (remember to reinstall your own version of GP_emu should you choose to do this).

<a name="Reconstruct an emulator"/>
### Reconstruct an emulator

When building an emulator, several files are saved at each step: an updated beliefs file and the inputs and outputs used in the construction of the emulator. The emulator can be rebuilt from these files without requiring another training run or build, since all the information is specified in these files. A minimal script would be:

```
import gp_emu as g

emul = g.setup("toy-sim_config_reconst")

g.plot(emul, [0,1],[2],[0.3], "mean")
```
where "toy-sim_config_reconst" contains the names of files generated from previously training an emulator e.g.:
```
beliefs toy-sim_beliefs-2f
inputs toy-sim_input-o0-2f
outputs toy-sim_output-o0-2f
```

*Be careful to specify the output correctly in the new beliefs file* - the updated output files from training an emulator will contain only a single column of outputs (the output for which the emulator was built, specified by ```output``` in the original beliefs file). For rebuilding an emulator, the beliefs file should specifiy that output 0 should be used (since we wish to use the first and only column of outputs in the updated output file).

*Be careful to specify the active inputs correctly in the new beliefs file* - the updated input files from training an emulator will contain only the active input dimensions (the inputs for which the emulator was build, specified by ```active``` in the original beliefs file), the updated inputs files will contain only these inputs. So inputs [0,2] of original inputs [0,1,2] will be indexed as [0,1] in the updated inputs file.

*Be especially careful with the tv_config option* - to reconstruct the emulator using all inputs points (without calling the ```train``` function) then the last value of tv_config must be 0.


<a name="Design Input Data"/>
## Design Input Data
See the following page for [MUCM's discussion on data design](http://mucm.aston.ac.uk/toolkit/index.php?page=AltCoreDesign.html)

To import this subpackage use something like this
```
import gp_emu.design_inputs as d
```
Currently, only an optimised Latin Hypercube design is included.

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
The design input points, output to _filename_, are suitable for reading by GP_emu.


<a name="Uncertainty and Sensitivity Analysis"/>
## Uncertainty and Sensitivity Analysis
See the following pages for MUCM's discussions on [uncertainty quantification](http://mucm.aston.ac.uk/toolkit/index.php?page=DiscUncertaintyAnalysis.html) and [sensitivity analysis](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadTopicSensitivityAnalysis.html).

The sensitivity subpackage can be used to perform uncertainty and sensitivity analysis. Currently, only a special case of an emulator with a Gaussian kernel and a linear mean function will work. The emulator inputs are assumed to be independant and normally distributed with mean m and variance v.

### Setup

Include the sensitivity subpackage as follows:
```
import gp_emu.sensitivity as s
```

A distribution for the inputs must be defined by a mean m and variance v for each input. These means and variances should be stored as a list e.g. for an emulator with 3 inputs with mean 0.50 and variance 0.02 for each input:

```
m = [0.50, 0.50, 0.50]
v = [0.02, 0.02, 0.02]
```

These lists and the emulator "emul" must then be passed to the a setup function which returns a Sensitivity class instance:

```
sens = s.setup(emul, m, v)
```

### Routines

#### Uncertainty

To perform uncertainty analysis to calculate, with respect to the emulator, the expection of the expection, the expection of the variance, and the variance of the expectation, use:
```
sens.uncertainty()
```

#### Sensitivity

To calculate sensitivity indices for each input, use:
```
sens.sensitivity()
```

#### Main Effect
To calculate and plot the main effects of each input, and optionally plot them (default ```plot = False```) use:
```
sens.main_effect(plot=True)
```
The number of points in the (scaled) input range 0.0 to 1.0 to use for plotting can be specified (default ```points = 100```):
```
sens.main_effect(plot=True, points = 200)
```
Extra optional arguments for the plot can also be chosen for the key, labels, the plot scale (useful for adjusting the exact placement of the key) and to use colours and linestyles suitable for black and white printing:
```
sens.main_effect(plot=True, customKey=['Na','K'], customLabels=['Model Inputs','Main Effect for dV/dt'], plotShrink=0.9, black_white=True)
```
An optional argument for the subset of inputs to be calculated/plotted can be provide (default is all inputs) e.g. to plot only input 0 and input 2:
```
sens.main_effect(plot=False, w=[0,2])
```

#### Interaction Effect

The interaction effect between two inputs {i,j} can be calculated and plotted with:
```
sens.interaction_effect(i, j)
```
Optional arguments can be supplied to specify the number of points used in each dimension (default=25, so the 2D plot will consist of 25*25 = 625 points) and labels for the x and y axes.
```
sens.interaction_effect(i, j, points = 25, customLabels=["input i", "input j"])
```

#### Total Effect Variance

The total effect variance for each input can be calculated with:
```
sens.totaleffectvariance()
```

#### Save results to file

To save calculated sensitivity results to file, use the to_file function:
```
sens.to_file("test_sense_file")
```

### Plot a sensitivity table
To plot a sensitivity table of the normalised sensitivities (sensitivity indices divided by the expectation of the variance) use
```
s.sense_table([sens,])
```
where the first argument is a list containing Sensitivity instances.

Optional arguments ```inputNames``` and ```outputNames``` for the column titles (inputs) and the row titles (emulator outputs) can be specified as lists.

An optional integer argument ```rowHeight``` (default 6) can adjust the height of the table rows.

By looping over different emulators (presumably built to emulate different outputs) and building up a list of Sensitivity instances, ```sense_table``` can be used to produce a table displaying the sensitivity of every output for every input.

```
sense_list = [ ]
for i in range(num_emulators):

    ... setup emulator ...

    ... setup sensitivity ...

    sense_list.append(sens)

s.sense_table(sense_list, [], [])
```


<a name="Examples"/>
## Examples

There are several examples in the top-level folder "examples".

<a name="Simple toy simulator"/>
### Simple toy simulator

#### toy-sim
To run a simple example, do
```
cd examples/toy-sim/
python emulator.py
```
The script emulator.py will attempt to build an emulator from the data found in toy-sim_input and toy-sim_output:
* toy-sim_input contains inputs generated from an optimised latin hypercube design
* toy-sim_output contains output generated by the script "toy-sim.py"

The script toy-sim.py is the 'toy simulation': it is simply a deterministic function performing some operations. To run, use:
```
python toy-sim.py toy-sim_input
```
or, for additive random noise from a normal distribution, with
```
python toy-sim.py toy-sim_input 0.25
```
where 0.25 is the amplitude multiplying the noise in this example.

Using the design_inputs subpackage, other input files (with more or less and/or more or less dimensions) can be generated to run this example. The function in toy-sim.py can be easily modified to accept higher dimensional input (4 inputs, 5 inputs etc.).

If adding noise to the toy simulation, then ```kernel gaussian() noise()``` could (should) be specified in the belief file.

#### toy-sim_reconstruct
This simple example demonstrates how to rebuild an emulator using files generated from previous training runs. Run with:
```
python emulator_reconst.py
```

<a name="Sensitivity Examples"/>
### Sensitivity examples

#### sensitivity_surfebm
This example demonstrates building an emulator and performing sensitivity analysis as in [this MUCM example](http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=ExamCoreGP2Dim.html).

#### sensitivity_multi_outputs
This example demonstrates building an emulators for simulations with multiple outputs. A separate emulator is built for each output, and by looping over different emulators it is possible to build a sensitivity table showing how all the outputs depend on all the inputs. Note that we need multiple config files and belief files specified.
