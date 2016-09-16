# TODO

## major dev
*Do I need to implement my new kernel addition idea to make this work, or will it work anyway?* Should work anyway, provided that the MUCM function calculates s^ and then sends it, along with the guess 'x', to the self.x_to_delta_and_sigma() function e.g. self.x_to_delta_and_sigma(x + s^) (since x will be one value shorter than the expected value). This means that I can leave the routine the same for the time being, I think.
Use analytic MUCM method (for sigma) fit for case of Gaussian kernel, since this should yield slightly better results. This means:
- optimal_full will spot gaussian kernel and limit the length of the *guess* to not include the sigma, so I may want to adjust the loop length...
- optimal_full needs to print to inform the user of this choice
- optimal_full needs to *send a different function to the optimzer* when calculating using the MUCM expression
- this different function can calculate s^ and save it to the kernel's sigma list - this way, common values can be calculated in one go rather than spread across separate routines.
- optimal_full must, in the MUCM case, adjust best_x to include the analytic value s^ on the end, so the other routines can function as normal.

Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS
code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D


## medium priority
Why are sigma and delta sent to each kernel - the kernels should know their own valuesi, right?
The sigma out the front should:
- leave beta unaffacted
-


GREAT IDEA: Could so s2(gaussian + (s2_noise/s2)noise + (s2_other/s2)other) so that single kernels would always be more efficient to calculate? We could return only the bracketed bit (in line with the MUCM definition of A) and keep the s2 out the front in the formula that use it... allows easy use of MUCM method too since the only difference is that we don't provide sigma to the loglikelihood and use the explicit formula instead.

Investigate accuracy and speed issues with inverting the correlation matrix, as discussed on MUCM: http://mucm.aston.ac.uk/toolkit/index.php?page=DiscBuildCoreGP.html
http://mucm.aston.ac.uk/MUCM/MUCMToolkit/index.php?page=DiscBuildCoreGP.html

Check that the remake functions are working efficiently - like, if we haven't just added V to T, then we don't need to remake the entire matrix.

Is delta naturally constrained to be positive?

provide extra loglikelihood fitting options for user, to help fit better emulators

## low priority
add official acknowledgements in the right places.

## documentation
links to the appropriate MUCM pages, which explain the methodology properly.
