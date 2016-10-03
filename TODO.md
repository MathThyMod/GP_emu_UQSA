# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Allow nugget to be fitted as an independant hyperparameter? Should be fairly easy to do.
Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS
code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - for now I have placed an exit() statement in the code to prevent use in 1D, but this needs fixing so it will work with simple 1D examples

## medium priority
Carefully check and compare the results of using different loglikelihood expressions, and crosscheck against external results

provide extra loglikelihood fitting options for user, to help fit better emulators - the new options that we added my be causing the fits to be slightly less precise - this needs investigating.

## low priority
tidy up the sensitivity routines.

The Hyperparameters class is almost redundant... used in the MUCM case... perhaps remove and just use beliefs? Or keep Hyperparameters as the 'working' set of these paramaters, so that beliefs is only accessed when we're updating our beliefs fully?

add official acknowledgements in the right places.

Some degeneracy in the constraints setting - when using type 'none' with constraints 'T', it still sets them...

IDEA: Could so s2(gaussian + (s2_noise/s2)noise + (s2_other/s2)other) so that single kernels would always be more efficient to calculate? We could return only the bracketed bit (in line with the MUCM definition of A) and keep the s2 out the front in the formula that use it... allows easy use of MUCM method too since the only difference is that we don't provide sigma to the loglikelihood and use the explicit formula instead.

## documentation
links to the appropriate MUCM pages, which explain the methodology properly.
