# TODO

## major dev ideas
Allow nugget to be fitted as an independant hyperparameter?
Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS
code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D

## medium priority
Investigate accuracy and speed issues with inverting the correlation matrix, as discussed on MUCM: http://mucm.aston.ac.uk/toolkit/index.php?page=DiscBuildCoreGP.html
http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=DiscBuildCoreGP.html

Check that the remake functions are working efficiently - like, if we haven't just added V to T, then we don't need to remake the entire matrix.

provide extra loglikelihood fitting options for user, to help fit better emulators

## low priority
add official acknowledgements in the right places.

IDEA: Could so s2(gaussian + (s2_noise/s2)noise + (s2_other/s2)other) so that single kernels would always be more efficient to calculate? We could return only the bracketed bit (in line with the MUCM definition of A) and keep the s2 out the front in the formula that use it... allows easy use of MUCM method too since the only difference is that we don't provide sigma to the loglikelihood and use the explicit formula instead.

## documentation
links to the appropriate MUCM pages, which explain the methodology properly.
