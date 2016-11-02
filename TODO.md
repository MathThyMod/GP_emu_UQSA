# TODO

## major dev ideas
History matching - determine inputs from given outputs.
Provide MUCM case 2 sensitivity routines.

## high priority -- BUGS
I need to allow the hyperparameters to be transformed depending on the kernel, since I don't want to restrict some delta to be positive.

MUCM llh probably won't work with the fix option at the moment.

Is the noise accounted for correctly in the posterior? NO. I've put in a fix whereby we don't build the covariance matrix with the noise in it for prediction. This needs to be better though.

Need to check that the mean function is being used correctly so that a zero mean can be set correctly (it may all already be fine) - having looked, it does seem to be ok.

## medium priority -- helpful features


## low priority -- small corrections

## documentation
add the climate example from GP4ML
