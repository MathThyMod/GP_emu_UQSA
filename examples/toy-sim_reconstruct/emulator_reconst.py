import gp_emu as g

emul = g.setup("toy-sim_config_reconst")
g.plot(emul, [0,1],[2],[0.3], "mean")
