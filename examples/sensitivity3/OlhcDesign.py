import gp_emu.design_inputs as d

#### configuration of design inputs
dim = 3
n = 100
N = 100
minmax = [ [0.0,1.0] , [0.0,1.0] , [0.0,1.0] ]
filename = "toy-sim_input"

#### call function to generate input file
d.optLatinHyperCube(dim, n, N, minmax, filename)
