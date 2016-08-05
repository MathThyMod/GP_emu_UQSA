# TODO

## major dev
Include sensitivity analysis routines - since MUCM nugget works differently to the standard noise term, how might we use their nugget gaussian for when we have noisy data? (may need to allow training on a nugget, in order to use the nugget for noise)

Ensure that the results I'm getting really are spot one, since they are slightly different from MUCM (although we use a different log likelihood formula... MUCM only guesses delta, we guess sigma and delta independantly)

## bugs or possible problems
code doesn't seem to work for 1D input...

mean not updated in final beliefs file (only getting updated internally)

the default sigma max is 10 - is this appropriate or too small for a defaul?

the code to generate the project directory and files doesn't work in python 2.7 due to the input command - fix this

## docs, examples, and tests
Provide meatier examples for user to run

Explain how to plot in 1D by supplying just the 1 dimension to plot
