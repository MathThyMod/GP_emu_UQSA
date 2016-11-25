# GP_emu_UQSA

```
=================    _(o> ====================
   GP_emu_UQSA    ¬(GP)      by Sam Coveney   
=================   ||    ====================
```
________

GP_emu_UQSA is a Python package to train a Gaussian Process Emulator and use it for uncertainty quantification and sensitivity analysis. It uses simple routines to encapsulate the [MUCM methodology](http://mucm.aston.ac.uk/toolkit/index.php?page=MetaHomePage.html).

To install GP_emu_UQSA, download/clone the package and run the following command inside the top directory:
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
GP_emu_UQSA uses the [methodology of MUCM](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadCoreGP.html) for building, training, and validating an emulator.

The user should create a project directory (separate from the GP_emu_UQSA download), and within it place a configuration file, a beliefs file, and an emulator script containing GP_emu_UQSA routines (the directory and these files can be created automatically - see [Create files automatically](#Create files automatically)). Separate inputs and outputs files should also be placed in this new directory.

The idea is to specify beliefs (e.g. form of mean function, values of hyperparameters) in the beliefs file, specify filenames and fitting options in the configuration file, and call GP_emu_UQSA routines with a script. The main script can be easily editted to specifiy using different configuration files and beliefs files, allowing the user's focus to be entirely on fitting the emulator.

Emulators should be trained on a design data e.g. optimized latin hypercube design. See [Design Input Data](#Design Input Data).

<a name="Main Script"/>
### Main Script
This script runs a series of functions in GP_emu_UQSA which automatically perform the main tasks outlined above. This allows flexibility for the user to create several different scripts for trying to fit an emulator to their data.

```
import gp_emu_uqsa as g

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

* The first argument is the emulator object.

* 2D plots: the second argument is a list of which input dimensions to use for the (x,y)-axes

* 1D plots: the second argument is a list of the input dimension to use for the x-axis

* The third argument is a list of which input dimensions to set to constant values

* The fourth argument is a list of these constant values.

* The fifth argument ```mean_or_var``` (optional, default ```"mean"```) specifies whether to plot the mean (```"mean"```) or the variance (```"var"```).

* The sixth argument ```customLabels``` (optional) is a list of strings for the axes labels.

* The seventh argument ```points``` (optional, default ```False```) specifies whether to plot the training points on the plot as well. This will only work for 1D plots of the mean.

* The eighth argument ```predict``` (optional, default ```True```) is to distinguish between 'prediction' (True) and 'estimation' (False). In prediction, the nugget (representing noise or jitter) will be included on the diagonal of the covariance k(x\*,x\*) of the posterior variance.

e.g. for a 1D plot of the variance with: input 0 varying from 0 to 1, inputs 1 and 2 held fixed at 0.10 and 0.20 respectively, x-label "input 0", plotted training points, and estimation (not prediction):
```
g.plot(emul, [0], [1,2], [0.10,0.20], "var", ["input 0"], True, False)
```

<a name="Config File"/>
### Config File
The configuration file does two things:

1. Specifies the names of the beliefs file and data files

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
  nugget_bounds [ ]
  fix [ ]
  tries 5
  constraints bounds
  ```

*The configuration file can be commented*, making it easy to test configurations without making a whole new config file e.g.
```
#delta_bounds [ [0.005 , 0.015] ]
delta_bounds [ [0.001 , 0.025] ]
```


#### tv_config
Specifies how the data is split into training and validation sets.

1. first value -- __10__ 0 2 -- how many sets to divide the data into (determines size of validation set - does not need to be a factor of the number of training points, remainders are added to the initial training set)

2. second value -- 10 __0__ 2 -- which V-set to initially validate against (currently, this should be set to zero; this mostly redundant feature is here to provide flexibility for implementing new features in the future)

3. third value -- 10 0 __2__ -- number of validation sets (determines number of training points)

| tv_config | data points | T points | V-set size | V sets  |
| ----------| ------------| -------- | ---------- | ------- |
| 10 0 2    | 200         | 160      | 20         | 2       |
| 4 0 1     | 100         | 75       | 25         | 1       |
| 10 0 1    | 100         | 90       | 10         | 1       |


#### delta_bounds and sigma_bounds and nugget_bounds
Sets bounds on the hyperparameters while fitting the emulator. These bounds will only be used if ```constraints bounds```, but *the constraints are always used as intervals in which to generate initial guesses for hyperparameter fitting*, therefore choosing good bounds is still important.

If the bounds are left empty, i.e. ```delta_bounds [ ]``` and ```sigma_bounds [ ]``` and ```nugget_bounds [ ]```, then they are automatically constructed, although these might not be suitable. The bounds on delta are based upon the range of inputs, and the bounds on sigma are based upon the largest output value, and the bounds on nugget are set to small values.

To set bounds, a list of lists must be constructed, the inner lists specifying the lower and upper range on each hyperparameter.


| input dimension | delta_bounds                           | sigma_bounds     | nugget_bounds        |
| --------------- | ------------                           | ------------     | ------------         |
| 3               | [ [0.0, 1.0], [0.1, 0.9], [0.2, 0.8] ] | [ [0.05, 5.0] ] | [ [0.00001, 0.010] ] |
| 1               | [ [0.0, 1.0] ]                         | [ [1.0, 10.0] ] | [ [0.001, 0.010] ] |


#### fitting options

* __tries__ : is how many times (integer) to try to fit the emulator for each round of training e.g. ``` tries 5 ```

* __constraints__ : which type of constraints to use, can be ```bounds``` (use the bounds in the configuration file), the default option ```standard``` (keep delta above a small value (0.001) for numerical stability), or ```none``` (unconstrained).


<a name="Beliefs File"/>
### Beliefs File
The beliefs file specifies beliefs about the data, namely which input dimensions are active, what mean function to use, and values of the hyperparameters (before training - this doesn't affect the training, except when using ```nugget_fix``` which will fix the nugget at the specified value).

```
active all
output 0
basis_str 1.0 x
basis_inf NA 0
beta 1.0 1.0
delta [ ]
sigma [ ]
nugget [ ]
fix_nugget T
mucm F
```

#### choosing inputs and outputs
The input dimensions to be used for the emulator are specified by ```active```. For all input dimensions use ```active all```, else list the input dimensions (index starts at 0) e.g. ```active 0 2```.

The output dimension to build the emulator for is specified by ```output```. Only a single index should be given e.g. ```output 2``` will use column '2' (technically the third column) of the output file.


#### the mean function
The specifications of ```basis_str``` and ```basis_inf``` define the mean function. ```basis_str``` defines functions and ```basis_inf``` defines which input dimension correspond to those functions. ```__beta__``` defines the mean function hyperparameters. The initial values of ```beta``` do not affect the emulator training, so they can be set to 1.0 for simplicity.

For mean function m(__x__) = b0
```
basis_str 1.0
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

#### the hyperparameters delta and sigma
The hyperparameters should be listed. Care should be taken to specify as many delta as required e.g. if the input has 5 dimensions then delta requires 5 values, but if only 3 input dimensions are active then only 3 delta should be given.
```
delta 1.0 1.0 1.0
sigma 1.0
```
The initial values do not affect how the emulator is trained.


#### the nugget
The nugget can be specified and fixed (T) or not (F) by
```
nugget 0.001
fix_nugget T
```
The nugget has several functions:
* a very small value can be used to provide numerical stability, as it can make the covariance matrix better conditioned. For this function, the nugget should be set as small as possible and fixed (not trained during the emulator fitting) so as to leave the fitting of delta and sigma as unaffected as possible (though there is always an effect).
* the nugget can be trained along with delta and sigma if ```fix_nugget F``` which allows the emulator to be trained on noisy data (bear in mind that if some inputs are 'turned off' but these inactive inputs vary across the training data set, that the data effectively becomes noisey). In this case, an estimate of the noise variance is printed during training.

#### mucm option
The mucm option provides a choice of loglikelihood methods:
* ```mucm F``` assumes vague priors and uses a 'standard' loglikelihood expression in which sigma is trained alongside delta (and nugget)
* ```mucm T``` assumes that the priors on sigma are inversely proportional to sigma, which allows sigma to be treated as an analytic function of delta (sigma is not independantly optimised)


<a name="Create files automatically"/>
### Create files automatically
A routine ```create_emulator_files()``` is provided to create a directory containing default belief, config, and main script files. This is to allow the user to easily set up different emulators.

It is simplest to run this function from an interactive python session as follows:
```
>>> import gp_emu_uqsa as g
>>> g.create_emulator_files()
```
The function will then prompt the user for input.


<a name="Fitting the emulator"/>
### Fitting the emulator
GP_emu_UQSA uses Scipy and Numpy routines for fitting the hyperparameters. The file \_emulatoroptimise.py contains the routines *differential\_evolution* and *minimize*, which can take additional arguments which GP_emu_UQSA (for simplicity) does not allow the user to specify at the moment. However, these additional arguments may make it easier to find the minimum of the negative loglikelihood function, and can easily be looked-up online and added to the code by the user (remember to reinstall your own version of GP_emu_UQSA should you choose to do this).

<a name="Reconstruct an emulator"/>
### Reconstruct an emulator

When building an emulator, several files are saved at each step: an updated beliefs file and the inputs and outputs used in the construction of the emulator. The emulator can be rebuilt from these files without requiring another training run or build, since all the information is specified in these files. A minimal script would be:

```
import gp_emu_uqsa as g

emul = g.setup("toy-sim_config_reconst")

g.plot(emul, [0,1],[2],[0.3], "mean")
```
where "toy-sim_config_reconst" contains the names of files generated from previously training an emulator e.g.:
```
beliefs toy-sim_beliefs-2f
inputs toy-sim_input-o0-2f
outputs toy-sim_output-o0-2f
```

#### Be careful

*Be careful to specify the output correctly in the new beliefs file* - the updated output files from training an emulator will contain only a single column of outputs (the output for which the emulator was built, specified by ```output``` in the original beliefs file). For rebuilding an emulator, the beliefs file should specifiy that output 0 should be used (since we wish to use the first and only column of outputs in the updated output file).

*Be careful to specify the active inputs correctly in the new beliefs file* - the updated input files from training an emulator will contain only the active input dimensions (the inputs for which the emulator was build, specified by ```active``` in the original beliefs file), the updated inputs files will contain only these inputs. So inputs [0,2] of original inputs [0,1,2] will be indexed as [0,1] in the updated inputs file.

*Be especially careful with the tv_config option* - to reconstruct the emulator using all inputs points (without calling the ```train``` function) then the last value of tv_config must be 0.


<a name="Design Input Data"/>
## Design Input Data
See the following page for [MUCM's discussion on data design](http://mucm.aston.ac.uk/toolkit/index.php?page=AltCoreDesign.html)

To import this subpackage use something like this
```
import gp_emu_uqsa.design_inputs as d
```
Currently, only an optimised Latin Hypercube design is included.

```
import gp_emu_uqsa.design_inputs as d

#### configuration of design inputs
dim = 2
n = 60
N = 200
minmax = [ [0.0,1.0] , [0.0,1.0] ]
filename = "toy-sim_input"

#### call function to generate input file
d.optLatinHyperCube(dim, n, N, minmax, filename)
```
The design input points, output to _filename_, are suitable for reading by GP_emu_UQSA.


<a name="Uncertainty and Sensitivity Analysis"/>
## Uncertainty and Sensitivity Analysis
See the following pages for MUCM's discussions on [uncertainty quantification](http://mucm.aston.ac.uk/toolkit/index.php?page=DiscUncertaintyAnalysis.html) and [sensitivity analysis](http://mucm.aston.ac.uk/toolkit/index.php?page=ThreadTopicSensitivityAnalysis.html).

The sensitivity subpackage can be used to perform uncertainty and sensitivity analysis. Currently, only a special case of an emulator with a Gaussian kernel and a linear mean function will work. The emulator inputs are assumed to be independant and normally distributed with mean m and variance v.

### Setup

Include the sensitivity subpackage as follows:
```
import gp_emu_uqsa.sensitivity as s
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
This example is within the toy-sim directory. It demonstrates how to rebuild an emulator using files generated from previous training runs. Run with:
```
python emulator_reconst.py
```

<a name="Sensitivity Examples"/>
### Sensitivity: sensitivity_surfebm
This example demonstrates building an emulator and performing sensitivity analysis as in [this MUCM example](http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=ExamCoreGP2Dim.html).

### Sensitivity: sensitivity_multi_outputs
This example demonstrates building an emulators for simulations with multiple outputs. A separate emulator is built for each output, and by looping over different emulators it is possible to build a sensitivity table showing how all the outputs depend on all the inputs. Note that we need multiple config files and belief files specified.
