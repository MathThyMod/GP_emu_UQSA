# TODO

## major dev
Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS
Big issue with rebuilding the emulator - the saved input and output files seem to be in scaled units! This means that rebuilding will only work if we set scaling to be false, right?

## medium priority
provide extra loglikelihood fitting options for user, to help fit better emulators - this could be in the form of suggesting the user modify the source files to provide extra arguments, or I could allow them to specify a whole array of additional arguments in the config file...

## low priority
add official acknowledgements in the right places.

code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D

## documentation
explain main effects options

explain how, when rebuilding, to probably use all the data points...

explain the datashuffle=True, scaleinputs=False options to the setup.

links to the appropriate MUCM pages, which explain the methodology properly.
