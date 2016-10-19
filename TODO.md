# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Provide loglikelihood options based on assumptions of priors (the MUCM and GP4ML use different loglikelihood functions)

## high priority -- BUGS

## medium priority -- helpful features
More efficient way to construct kernel: _I should have a function that returns the distance matrix - if already calculated before then return the matrix, otherwise calculate it. Becuase for a particular training set we only need to calculate this quantity once, surely???_

Check that rational quadratic is correct (i.e. no deltas, only sigmas)

Why does GP4ML periodic kernel differ from the form in the Kernel Cookbook ??? Wiki is on side of GP4ML...

Option to plot the training points on the plots

Make the mean effects plots have enough different colours

## low priority -- small corrections


## documentation
add the climate example from GP4ML
