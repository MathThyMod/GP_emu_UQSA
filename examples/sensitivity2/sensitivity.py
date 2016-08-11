import gp_emu as g
import gp_emu.sensitivity as s

#### set up everything - config, emulator
conf = g.config("toysim3D_config")
emul = g.setup(conf, datashuffle=True, scaleinputs=False)

#### repeat train and validate, then retrain into final emulator
g.final_build(emul, conf, auto=True)

#### set up sensitivity analysis - requires the emulator
m = [0.50, 0.50, 0.50]
v = [0.02, 0.02, 0.02]
sens = s.setup(emul, "case2", m, v)
#sens.main_effect()
sens.sensitivity()
sens.totaleffectvariance()
