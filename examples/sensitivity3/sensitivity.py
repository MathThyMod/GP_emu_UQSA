import gp_emu as g
import gp_emu.sensitivity as s

sense_list = []

## loop over different emulators
for i in range(2):
    #### set up everything - config, emulator
    conf = g.config("toysim3D_config")
    emul = g.setup(conf, datashuffle=True, scaleinputs=False)

    #### repeat train and validate, then retrain into final emul[i]ator
    g.final_build(emul, conf, auto=True)

    #### set up sensitivity analysis - requires the emul[i]ator
    m = [0.50, 0.50, 0.50]
    v = [0.02, 0.02, 0.02]
    sens = s.setup(emul, "case2", m, v)
    sens.uncertainty()
    sens.sensitivity()
    sens.main_effect(plot=False)
    sens.to_file("test_sense_file")
    sense_list.append(sens)
    #sens.interaction_effect(0, 1)
    #sens.totaleffectvariance()

## make the sense table
s.sense_table(sense_list, [], [])
