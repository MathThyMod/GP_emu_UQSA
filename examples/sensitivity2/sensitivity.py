import gp_emu as g
import gp_emu.sensitivity as s

#### set up everything - config, emulator
conf = g.config("toysim3D_config")
emul = g.setup(conf, datashuffle=True, scaleinputs=True)

#### repeat train and validate, then retrain into final emulator
g.final_build(emul, conf, auto=True)

#### set up sensitivity analysis - requires the emulator
m = [0.50, 0.50, 0.50]
v = [0.02, 0.02, 0.02]
sens = s.setup(emul, "case2", m, v)
sens.uncertainty()
sens.sensitivity()
sens.main_effect(plot=False)
sens.to_file("test_sense_file")
#sens.interaction_effect(0, 1)
#sens.totaleffectvariance()
