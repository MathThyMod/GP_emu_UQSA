# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Provide MUCM case 2 sensitivity routines.

## high priority -- BUGS
Not sure the diagnositcs are working correctly with kernels with experimental noise. I may be that when the validation set includes noise that I do want K(X*,X*) to include noise as well.

Is the noise accounted for correctly in the posterior? NO. I've put in a fix whereby we don't build the covariance matrix with the noise in it for prediction. This needs to be better though.

Need to check that the mean function is being used correctly so that a zero mean can be set correctly (it may all already be fine) - having looked, it does seem to be ok.

## medium priority -- helpful features
If not PSD, maybe try the next initial guess until we've exhausted the guesses.

## low priority -- small corrections

## documentation
add the climate example from GP4ML
