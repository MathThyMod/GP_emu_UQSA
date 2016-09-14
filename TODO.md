# TODO

## major dev
Use analytic MUCM method (for sigma) fit for case of Gaussian kernel, since this should yield slightly better results.

Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS
code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D


## medium priority
We should only add the kernel matrices together iafter we've done everything in squareform, else waste of time - this means we need to be careful, since squareform won't store the diagonal parts, so we'll have to add the squareforms and then add all the diagonal parts -- DONE! but needs checking

Can we do the exponential of the UT (not squareform) and then make squareform? (Same idea with nugget?) -- DONE! but needs checking

Investigate accuracy and speed issues with inverting the correlation matrix, as discussed on MUCM: http://mucm.aston.ac.uk/toolkit/index.php?page=DiscBuildCoreGP.html

Investigate separating sigma from A as much as possible, to reduce number of computations needed.

Precompute more stuff in advance in the LLH.

Check that the remake functions are working efficiently - like, if we haven't just added V to T, then we don't need to remake the entire matrix.

Is delta natural constrained to be positive?

Check that the nugget loops are only running when we actually have a non-zero nugget.


provide extra loglikelihood fitting options for user, to help fit better emulators

## low priority
add official acknowledgements in the right places.

## documentation
links to the appropriate MUCM pages, which explain the methodology properly.
