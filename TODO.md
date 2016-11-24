# TODO

## major dev ideas

## high priority -- BUGS
the fix_mean option probably doesn't work because the LLH actually implicitly calculates beta values internally - the fix mean value may work with a fixed zero mean if the basis function is set to 0, but this needs carefully checking. One solution is just to remove the fixed mean idea and just allow for a constant mean to be fitted.

## medium priority -- helpful features
Find a faster optimised latin hypercube method, if possible

## low priority -- small corrections

## documentation
Update the documentation GP_emu_UQSA is very simple again.
