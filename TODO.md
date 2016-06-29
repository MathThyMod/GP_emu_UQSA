# TODO

## major dev
Include sensitivity analysis routines

## bugs (known or possible)
Sometimes get negative ISEs - is this just wrong? Could just be a numerical issue (i.e not a code bug)

## improvements
Possibly place configuation info into the main script

Possibly place kernel into the beliefs script? Is this even possible?

Avoid doing inversion of A at all - this is very expensive. Instead solve Ax=y for x. by passing A and y to a function - A will be the covariance matrix and y will be whatever the inverse of the covariance matric is multiplying. The function will then return x.

Provide file output of posterior (so that we have more than just a plot) -- this file would be huge though... maybe better to reconstruct emulator with GP_emu

## examples and tests
include an explanation of the toysim example?

recommended to place all the files in a single directory

Provide meatier examples for user to run
