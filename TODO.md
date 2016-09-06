# TODO

## major dev
Allow nugget to be fitted as an independant hyperparameter?

Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

Get the Interaction Effects working properly.

Get the totaleffectvariance working properly.

## high priority -- BUGS

## medium priority
provide extra loglikelihood fitting options for user, to help fit better emulators - this could be in the form of suggesting the user modify the source files to provide extra arguments, or I could allow them to specify a whole array of additional arguments in the config file.

## low priority
create emulator files should have a sensitivity option

append ".py" to the emulator script that's on auto create

main effects plots - option for unscaled axes, xy labels, and postscript options.

sensitivity table - decimal places options.

add official acknowledgements in the right places.

code doesn't work for 1D input anymore because of syntax like x[:,i] which causes 1D inputs to be treated like 2D inputs - probably not worth implementing, so for now I have placed an exit() statement in the code to prevent use in 1D

## documentation
explain main effects options.

explain the datashuffle=True, scaleinputs=False options to the setup.

links to the appropriate MUCM pages, which explain the methodology properly.
