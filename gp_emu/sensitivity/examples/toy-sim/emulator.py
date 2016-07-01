import gp_emu as g
import gp_emu.sensitivity as s

#### set up everything - config, emulator
conf = g.config("toy-sim_config")
emul = g.setup(conf)

#### repeat train and validate, then retrain into final emulator
g.training_loop(emul, conf, auto=True)

#### set up sensitivity analysis - requires the emulator
s.setup(emul)

