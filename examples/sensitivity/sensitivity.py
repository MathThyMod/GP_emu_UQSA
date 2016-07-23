import gp_emu as g
import gp_emu.sensitivity as s

#### set up everything - config, emulator
conf = g.config("surfebm_config")
emul = g.setup(conf)

#### repeat train and validate, then retrain into final emulator
g.final_build(emul, conf, auto=True)

#### set up sensitivity analysis - requires the emulator
s.setup(emul, "case2")

