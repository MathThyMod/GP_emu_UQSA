from __future__ import print_function
from builtins import input
import os as __os

def create_emulator_files():
    print("This will create a new directory containing files for GP_emu_UQSA")
    name = input("Name of emulator: ")
    if not __os.path.exists(name):
        __os.makedirs(name)
    else:
        print("That directory already exists, aborting")
        __os.sys.exit()

    inputs = input("Number of inputs: ")
    inputs=int(inputs)

    beliefs = name + "_beliefs"
    print("beliefs:" , beliefs)
    config = name + "_config"
    print("config:" , config)
    emulator = name + "_emulator" + ".py"
    print("emulator:" , emulator)

    basis_str=""
    basis_inf=""
    beta=""
    delta=""
    for i in range(0,inputs):
        basis_str = basis_str + " x"
        basis_inf = basis_inf + " " + str(i)
        beta = beta + " 1.0"
        delta = delta + " 1.0"


    print("Creating beliefs file...") 
    with open( __os.path.join(name,beliefs), 'w' ) as bf:
        bf.write("active all\n")
        bf.write("output 0\n")
        bf.write("basis_str 1.0" + basis_str + "\n")
        bf.write("basis_inf NA" + basis_inf + "\n")
        bf.write("beta 1.0" + beta + "\n")
        bf.write("fix_mean F\n")
        bf.write("delta" + delta + "\n")
        bf.write("sigma 1.0\n")
        bf.write("nugget 0.0\n")
        bf.write("fix_nugget T\n")
        bf.write("mucm F\n")

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
        cf.write("nugget_bounds [ ]\n")
        cf.write("tries 1\n")
        cf.write("constraints bounds\n")

    print("Inputs and outputs files named",inputs_filename,"&",outputs_filename,"in the config file. Remember to include the input and output files in the new directory.")

    
    sens = input("Include sensitivity routines? y/[n]: ")
    if sens == "y":
        sensScript = True
    else:
        sensScript = False

    if sensScript != True:
        print("Creating emulator script file...")
        with open( __os.path.join(name,emulator), 'w' ) as ef:
            ef.write("import gp_emu_uqsa as g\n")
            ef.write("\n")
            ef.write("emul = g.setup(\""+config+"\")\n")
            ef.write("g.train(emul, auto=True)\n")
    else:
        print("Creating emulator + sensitivity script file...")
        with open( __os.path.join(name,emulator), 'w' ) as ef:
            ef.write("import gp_emu_uqsa as g\n")
            ef.write("import gp_emu_uqsa.sensitivity as s\n")
            ef.write("\n")
            ef.write("emul = g.setup(\""+config+"\")\n")
            ef.write("g.train(emul, auto=True)\n")
            ef.write("\n")
            m = [0.50 for i in range(0,inputs)]
            v = [0.02 for i in range(0,inputs)]
            ef.write("m = " + str(m) + "\n")
            ef.write("v = " + str(v) + "\n")
            
            ef.write("sens = s.setup(emul, m, v)\n")
            ef.write("sens.uncertainty()\n")
            ef.write("sens.sensitivity()\n")
            ef.write("sens.main_effect(plot=True)\n")
            ef.write("sens.to_file(\"sense_file\")\n")
            ef.write("sens.interaction_effect(0, 1)\n")
            ef.write("sens.totaleffectvariance()\n")
