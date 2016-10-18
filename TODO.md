# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Provide loglikelihood options based on assumptions of priors (the MUCM and GP4ML use different loglikelihood functions)

## high priority -- BUGS

## medium priority -- helpful features
Add locally_periodic kernel

Check that rational quadratic is correct (i.e. no deltas, only sigmas)

Why does GP4ML periodic kernel differ from the form in the Kernel Cookbook ???

Option to plot the training points on the plots

## low priority -- small corrections


## documentation
add the climate example from GP4ML

print hyperparameter info when kernel is constructed

explain the options for the optimisation a little better (stochastic is bounded so will always use constraints, and if stochastic is false then constaints option will choose to use a gradient optimiser as constrained or not)
