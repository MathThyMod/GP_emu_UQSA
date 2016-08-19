# TODO

## major dev
Allow MUCM nugget to be fitted as an independant hyperparameter and include MUCM nugget in sensitivity calculations? Alternative is to implement sensitivity analysis that allows general kernels (rather than just the Gaussian kernel).

## bugs
code doesn't seem to work for 1D input anymore because of syntax like x[:,i]

add nugget to the sensitivity analysis, since this is very simple

## low priority
name of input and output files should have the output number appeneded

reduce number of saved input & output files - just have one for the last build we did, maybe?

let the plotting options fixed_vals be specified in unscaled units?

## docs, examples, and tests
Provide meatier examples for user to run and documentation explaining these.
