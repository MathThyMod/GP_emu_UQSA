# TODO

## major dev
Include sensitivity analysis routines - since MUCM nugget works differently to the standard noise term, how might we use their nugget gaussian for when we have noisy data?

## bugs (known or possible)
Sometimes get negative ISEs - is this just wrong? Could just be a numerical issue (i.e not a code bug)

## improvements
Inverses removed from bulk of program - still need removing in a few place (where the calculations are done very few times, so it is unlikely to improve performance) - remove these direct inverses after testing the current improved code

Provide file output of posterior (so that we have more than just a plot) -- this file would be huge though... maybe better to reconstruct emulator with GP_emu

Possibly place configuation info into the main script

## docs, examples, and tests
include some explanation of the toysim example e.g. what the underlying simulation function actually is, and how to create different outputs using the provided toy-sim simulation

recommended to place all the files in a single directory

Provide meatier examples for user to run
