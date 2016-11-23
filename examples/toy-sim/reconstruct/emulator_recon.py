import gp_emu_uqsa as g

print("Reconstruct a previously built emulator - no training needed.")

#### set up the emulator
emul = g.setup("toy-sim_config_recon")

#### see full prediction, plot "mean" or "var"
g.plot(emul, [0],[1],[0.3], "mean")
g.plot(emul, [0,1],[2],[0.3], "mean")
