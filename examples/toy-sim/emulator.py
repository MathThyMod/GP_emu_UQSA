import gp_emu as g

#### set up the emulator
emul = g.setup("toy-sim_config")

#### training and validation
g.train(emul, auto=True)

#### see full prediction, plot "mean" or "var"
g.plot(emul, [0],[1],[0.3], "mean")
g.plot(emul, [0,1],[2],[0.3], "mean")
