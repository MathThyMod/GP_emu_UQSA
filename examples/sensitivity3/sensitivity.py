import gp_emu as g
import gp_emu.sensitivity as s

sense_list = []

## loop over different emulators
for i in range(2):
    #### set up everything - config, emulator
    conf = g.config("toysim3D_config")
    emul = g.setup(conf, datashuffle=True, scaleinputs=True)

    #### repeat train and validate, then retrain into final emulator
    g.final_build(emul, conf, auto=True)

    #### set up sensitivity analysis - requires the emulator
    m = [0.50, 0.50, 0.50]
    v = [0.02, 0.02, 0.02]
    
    ### FIGURE OUT WHY THERE'S A PROBLEM FITTING THIS EMULATOR...

    sens = s.setup(emul, "case2", m, v)
    sens.uncertainty()
    sens.sensitivity()
    #sens.main_effect(plot=True) ## PROVIDE OPTION FOR NUMBER OF POINTS
    sens.to_file("test_sense_file")
    sense_list.append(sens)
    #sens.interaction_effect(0, 1)
    #sens.totaleffectvariance()

## make the sense table
s.sense_table(sense_list, [], [])
