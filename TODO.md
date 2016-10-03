# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Allow nugget to be fitted as an independant hyperparameter? Should be fairly easy to do.
Include "special case 2" for the sensitivity (so that we can generalise the mean function)?

## high priority -- BUGS

## medium priority
Carefully check and compare the results of using different loglikelihood expressions, and crosscheck against external results

provide extra loglikelihood fitting options for user, to help fit better emulators - the new options that we added my be causing the fits to be slightly less precise - this needs investigating.

## low priority
make routine in design_inputs subpackage to create a design_inputs script automatically and/or change the design_inputs routine to ask user for inputs from keyboard - this is even simpler...

Some degeneracy in the constraints setting - when using type 'none' with constraints 'T', it still sets them - fix?

Remove the arbitrary restriction of decimal places in the final input and output files

## documentation
comment all of the functions and bits of the functions correctly

links to the appropriate MUCM pages, which explain the methodology properly
