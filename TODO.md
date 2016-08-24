# TODO

## major dev
Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## bugs

## low priority
check for more speed ups in the sentitivity

code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D

## documentation
