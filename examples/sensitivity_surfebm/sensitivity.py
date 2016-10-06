import gp_emu as g
import gp_emu.sensitivity as s

emul = g.setup("surfebm_config", datashuffle=True, scaleinputs=False)
g.train(emul, auto=True)

if True:
    #### set up sensitivity analysis - requires the emulator
    m = [0.50, 0.50]
    v = [0.02, 0.02]
    sens = s.setup(emul, m, v)
    sens.uncertainty()
    sens.sensitivity()
    sens.main_effect(plot=True, points=100, customKey=["sam1", "sam2"], plotShrink=0.8)
    sens.to_file("test_sense_file")
    sens.interaction_effect(0, 1)
    sens.totaleffectvariance()

    sense_list = [sens, ]

    s.sense_table(sense_list, ["input 0", "input 1"], ["output 0"])
    #s.sense_table(sense_list, [], [])
    #s.sense_table(sense_list, [], [], 2)
