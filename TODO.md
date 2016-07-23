# TODO

## major dev
Include sensitivity analysis routines - since MUCM nugget works differently to the standard noise term, how might we use their nugget gaussian for when we have noisy data? (may need to allow training on a nugget, in order to use the nugget for noise)

Ensure that the results I'm getting really are spot one, since they are slightly different from MUCM (although we use a different log likelihood formula... MUCM only guesses delta, we guess sigma and delta independantly)

## docs, examples, and tests
Provide meatier examples for user to run
