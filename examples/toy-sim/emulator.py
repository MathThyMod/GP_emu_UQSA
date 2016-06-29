import gp_emu as g

#### set up everything - config, kernel, emulator
conf = g.config_file("toy-sim_config")
K = g.gaussian() + g.noise()
emul = g.setup(conf, K)

#### repeat train and validate, then retrain into final emulator
g.training_loop(emul, conf, auto=True)
g.final_build(emul, conf, auto=False)

#### see full prediction, plot "mean" or "var"
g.plot(emul, [0,1],[2],[0.65], "mean")
