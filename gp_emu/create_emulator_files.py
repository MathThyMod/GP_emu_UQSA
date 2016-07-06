import os as __os

def create_emulator_files():
    name = input("Name of emulator: ")
    if not __os.path.exists(name):
        __os.makedirs(name)
    else:
        print("That folder already exists, aborting")
        __os.sys.exit()

    inputs = input("Number of inputs: ")
    inputs=int(inputs)

    beliefs = name + "_beliefs"
    print("beliefs:" , beliefs)
    config = name + "_config"
    print("config:" , config)
    emulator = name + "_emulator"
    print("emulator:" , emulator)

    basis_str=""
    basis_inf=""
    beta=""
    for i in range(0,inputs):
        basis_str = basis_str + " x"
        basis_inf = basis_inf + " " + str(i)
        beta = beta + " 1.0"


    print("Creating beliefs file...") 
    with open( __os.path.join(name,beliefs), 'w' ) as bf:
        bf.write("active all\n")
        bf.write("output 0\n")
        bf.write("basis_str 1.0" + basis_str + "\n")
        bf.write("basis_inf NA" + basis_inf + "\n")
        bf.write("beta 1.0" + beta + "\n")
        bf.write("fix_mean F\n")
        bf.write("kernel gaussian()\n")
        bf.write("delta [ ]\n")
        bf.write("sigma [ ]\n")

    inputs_filename=name + "_inputs"
    outputs_filename=name + "_outputs"
    print("Creating config file...") 
    with open( __os.path.join(name,config), 'w' ) as cf:
        cf.write("beliefs " + beliefs + "\n")
        cf.write("inputs " + inputs_filename+"\n")
        cf.write("outputs " + outputs_filename+"\n")
        cf.write("tv_config 10 0 1\n")
        cf.write("delta_bounds [ ]\n")
        cf.write("sigma_bounds [ ]\n")
        cf.write("tries 1\n")
        cf.write("constraints T\n")
        cf.write("stochastic F\n")
        cf.write("constraints_type bounds\n")

    print("Inputs and outputs files named",inputs_filename,"&",outputs_filename,"in the config file. Remember to include the input and output files in the new directory (and rename them in the config file if necessary)")

    print("Creating emulator script file...")
    with open( __os.path.join(name,emulator), 'w' ) as ef:
        ef.write("import gp_emu as g\n")
        ef.write("\n")
        ef.write("conf = g.config("+config+")\n")
        ef.write("emul = g.setup(conf)\n")
        ef.write("g.training_loop(emul, conf, auto=True)\n")
        ef.write("g.final_build(emul, conf, auto=True)\n")
