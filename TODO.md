# TODO

## major dev ideas
History matching - determine inputs from given outputs.

## high priority -- BUGS
the examples (and documentation) need redoing now that the useless config function has been removed - do this after other changes to the names of the other routines

## medium priority
Carefully check and compare the results of using different loglikelihood expressions, and crosscheck against external results -- the MUCM and GP4ML use different assumptions on the priors it seems

## low priority
it's possible to divide by zero in Mahalanobis and llh if too few point... - warn that we must train with points>6

the autoconfigure kernel function doesn't use any delta for noise, which is fine, but is there a better automatic way?

config now part of the emulator class - should allow a lot of the code to be simplified

when rebuilding, we always have to specifiy the belief_file output back to 0, which is annoying... what to do?

rebuilding it tells user when V added to T at a weird time

## documentation
comment all of the functions and bits of the functions correctly

links to the appropriate MUCM pages, which explain the methodology properly
