import gp_emu as g

### TODO
#### *** avoid doing inversion of A at all - expensive! But this probably won't matter unless we've got loads of points, which don't... and probably won't for physics-type projects...***

#### ***IMPORTANT*** SHOULD THE NUGGET STILL BE IN THE POST (THE I NEQ J PARTS, OBVIOUSLY) ??? NEED TO FIGURE OUT AND FIX

#### set up everything - config, kernel, emulator
conf = g.config_file("toy-sim/toy-sim_config")
K = g.Gaussian() + g.Noise()
emul = g.setup(conf, K)

#### repeat train and validate, then retrain into final emulator
g.training_loop(emul, conf)
g.final_build(emul, conf)

#### see full prediction, plot "mean" or "var"
g.plot(emul, [0,1],[2],[0.65], "mean")
