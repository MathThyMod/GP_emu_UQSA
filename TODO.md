# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Provide MUCM case 2 sensitivity routines.

## high priority -- BUGS
Need to check that the mean function is being used correctly so that a zero mean can be set correctly (it may all already be fine)

## medium priority -- helpful features
Plots should work in whatever input range is provided, in case data hasn't been scaled into the range 0 to 1, so this needs fixing.

If we haven't scaled the data into the range 0 to 1 then the initial guesses are going to be very poor - perhaps the default bounds should be the minmax of the full training data or something like that?


## low priority -- small corrections
Option to plot the training points on the plots?

## documentation
add the climate example from GP4ML
