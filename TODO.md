# TODO

## misc.
what to do with the LHC stuff - this is a useful tool, so how should it be used in the package? It should probably be turned into a submodule, and the config file for it should just be part of a user script which calls a function from this module

Hide print_function from the user interace

## bugs (known or possible)
Correct naming of last build file - probably append an 'f' to the file name, to distinguish the a training run including validation data in training (append f) and the training run using the validation data for diagnostics

Should the Nuggest be in the posterior, or not? This is important or the nugget won't function properly.

Sometimes get negative ISEs - is this just wrong?

## improvements
Is there a need for a separate final build function, or can this be combined into the training loop function somehow?

Avoid doing inversion of A at all - this is very expensive. Instead solve Ax=y for x. by passing A and y to a function - A will be the covariance matrix and y will be whatever the inverse of the covariance matric is multiplying. The function will then return x.

## examples and tests
update README.md with an overview of the project, explaining the main script and the beliefs and config files

include an explanation of the toysim example?

recommended to place all the files in a single directory

Provide meatier examples to run
