import gp_emu as g
import gp_emu.sensitivity as s

#### set up everything - config, emulator
conf = g.config("surfebm_config")
emul = g.setup(conf, datashuffle=True, scaleinputs=False)

#### repeat train and validate, then retrain into final emulator
g.final_build(emul, conf, auto=True)

if False:
    #### set up sensitivity analysis - requires the emulator
    m = [0.50, 0.50]
    v = [0.02, 0.02]
    sens = s.setup(emul, m, v)
    sens.uncertainty()
    sens.sensitivity()
    sens.main_effect(plot=True, points=20)
    sens.to_file("test_sense_file")
    #sens.interaction_effect(0, 1)
    #sens.totaleffectvariance()

    sense_list = [sens, ]

    #s.sense_table(sense_list, ["input 0", "input 1"], ["output 0"])
    #s.sense_table(sense_list, [], [])
    #s.sense_table(sense_list, [], [], 2)
