# TODO

## major dev ideas
Case 1 sensitivity routines, so that we don't need a linear mean function.
History matching - determine inputs from given outputs.
Provide loglikelihood options based on assumptions of priors (the MUCM and GP4ML use different loglikelihood functions)

## high priority -- BUGS
Why doesn't the answer change when setting to a fixed zero mean?

## medium priority -- helpful features
Should the Gaussian kernel have the 2 in the denominator or not? I've put it for now so I can test fitting the Mauna Loa climate data...

Plots should work in whatever input range is provided, in case data hasn't been scaled into the range 0 to 1, so this needs fixing.

If we haven't scaled the data into the range 0 to 1 then the initial guesses are going to be very poor - perhaps the default bounds should be the minmax of the full training data or something like that?


## low priority -- small corrections
Option to plot the training points on the plots?

## documentation
add the climate example from GP4ML
