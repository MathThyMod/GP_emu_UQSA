# TODO

## major dev
Allow MUCM nugget to be fitted as an independant hyperparameter and iclude MUCM nugget in sensitivity calculations? Alternative is to implement sensitivity analysis that allows general kernels (rather than just the Gaussian kernel).

## bugs or possible problems
set a maximum number of iterations to perform for optimizing, in case of a bad guess for the stochastic method...

let the plotting options fixed_vals be specified in unscaled units, so it's easier for the user?

code doesn't seem to work for 1D input...

the code to generate the project directory and files doesn't work in python 2.7 due to the input command - fix this

## docs, examples, and tests
Provide meatier examples for user to run

Explain how to plot in 1D by supplying just the 1 dimension to plot
