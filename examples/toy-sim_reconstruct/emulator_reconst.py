import gp_emu as g

conf = g.config("toy-sim_config_reconst")
emul = g.setup(conf)
g.plot(emul, [0,1],[2],[0.3], "mean")
