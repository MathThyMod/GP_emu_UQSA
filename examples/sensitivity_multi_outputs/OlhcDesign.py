import gp_emu.design_inputs as d

#### configuration of design inputs
dim = 3
n = 100
N = 200
minmax = [ [0.0,1.0] , [0.0,1.0] , [0.0,1.0] ]
filename = "toysim3D_input"

#### call function to generate input file
d.optLatinHyperCube(dim, n, N, minmax, filename)
