# TODO

## major dev
Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS


## medium priority
Investigate accuracy and speed issues with inverting the correlation matrix, as discussed on MUCM: http://mucm.aston.ac.uk/toolkit/index.php?page=DiscBuildCoreGP.html

provide extra loglikelihood fitting options for user, to help fit better emulators

## low priority
add official acknowledgements in the right places.

code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D

## documentation
links to the appropriate MUCM pages, which explain the methodology properly.
