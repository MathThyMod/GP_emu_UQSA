from __future__ import print_function
from builtins import input
import numpy as np
from scipy import linalg


class Emulator:
    """Keeps instances of other classes together for convenience."""
    def __init__(
            self, config, beliefs, par, basis, tv_conf, all_data,
            training, validation, post, opt_T, K):
        self.config = config
        self.beliefs = beliefs
        self.par = par
        self.basis = basis
        self.tv_conf = tv_conf
        self.all_data = all_data
        self.training = training
        self.validation = validation
        self.post = post
        self.opt_T = opt_T
        self.K = K

### configuration file, naming all info that isn't a belief
class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.read_file()
        self.set_values()

    def read_file(self):
        print("*** Reading config file:", self.config_file ,"***")
        self.config = {}
        try:
            with open(self.config_file, 'r') as f:
                for line in f:
                    (key, val) = line.split(' ',1)
                    self.config[key] = val
        except OSError as e:
            print("ERROR: Problem reading file.")
            exit()


        # check for presence of all required keywords
        for i in ['beliefs', 'inputs', 'outputs', 'tv_config',\
                  'delta_bounds', 'sigma_bounds',\
                  'tries', 'constraints', 'stochastic', 'constraints_type']:
            try:
                self.config[i]
            except KeyError as e:
                print("WARNING: \"", i, "\" specification is missing")
                exit()


    def set_values(self):

        self.beliefs = str(self.config['beliefs']).strip()
        self.inputs = str(self.config['inputs']).strip()
        self.outputs = str(self.config['outputs']).strip()

        self.tv_config = tuple(str(self.config['tv_config']).strip().split(' '))
        self.tv_config =\
            [int(self.tv_config[i]) for i in range(0, len(self.tv_config))]
        if len(self.tv_config) != 3:
            print("WARNING: tv_config requires 3 entries.")
            exit()
        print("T-V config:", self.tv_config)

        delta_bounds_t = eval( str(self.config['delta_bounds']).strip() )
        sigma_bounds_t = eval( str(self.config['sigma_bounds']).strip() )
        self.bounds = tuple(delta_bounds_t + sigma_bounds_t)

        self.tries = int(str(self.config['tries']).strip())
        print("number of tries for optimum:" , self.tries)

        constraints = str(self.config['constraints']).strip()
        if constraints == 'T':
            self.constraints = True
        else:
            if constraints == 'F':
                self.constraints = False
            else:
                print("WARNING: constraints must be T or F")
                exit()
        print("constraints:" , self.constraints)

        stochastic = str(self.config['stochastic']).strip()
        if stochastic == 'T':
            self.stochastic = True
        else:
            if stochastic == 'F':
                self.stochastic = False
            else:
                print("WARNING: stochastic must be T or F")
                exit()
        print("stochastic:" , self.stochastic)

        self.constraints_type = str(self.config['constraints_type']).strip()
        print("constraints_type:", self.constraints_type)


### gathers all the beliefs from the specified belief file
class Beliefs:
    def __init__(self,beliefs_file):
        self.beliefs_file=beliefs_file
        self.read_file()
        self.set_values()
        
    def read_file(self):
        ## read into a dictionary
        print("\n*** Reading beliefs file:" , self.beliefs_file , "***")
        self.beliefs = {}
        try:
            with open(self.beliefs_file, 'r') as f:
                for line in f:
                    (key, val) = line.split(' ',1)
                    self.beliefs[key] = val
        except OSError as e:
            print("ERROR: Problem reading file.")
            exit()

        # check for presence of all required keywords
        for i in ['active', 'output', 'basis_str', 'basis_inf', 'beta',\
                  'fix_mean', 'kernel', 'delta', 'sigma']:
            try:
                self.beliefs[i]
            except KeyError as e:
                print("WARNING: \"", i, "\" specification is missing")
                exit()

    def set_values(self):

        # which input dimensions to use
        self.active = []
        active = str(self.beliefs['active']).strip().split(' ')[0:]
        if active[0] == "all":
            self.active = []
        else:
            active = list(active)
            self.active = [int(active[i]) for i in range(0, len(self.active))]
        print("active:", self.active)

        # which output dimension to use
        self.output = int( str(self.beliefs['output']).strip().split(' ')[0] )
        print("output:",self.output)

        # mean function specifications
        self.basis_str = (str(self.beliefs['basis_str']).strip().split(' '))
        self.basis_inf =\
          [int(i) for i in str(self.beliefs['basis_inf']).strip().split(' ')[1:]]
        self.beta =\
          [float(i) for i in (str(self.beliefs['beta']).strip().split(' '))]

        # check that the mean function specifications are correct
        if len(self.basis_str) != len(self.basis_inf) + 1:
            print("WARNING: basis_str & basis_inf need an equal number of "
                  "entires, including redundant first entry of basis_inf.")
            exit()
        if len(self.basis_str) != len(self.beta):
            print("WARNING: basis_str & beta need an equal number of entries.")
            exit()

        self.fix_mean = str(self.beliefs['fix_mean']).strip().split(' ')[0]
        self.kernel = str(self.beliefs['kernel']).strip().split(' ')
        self.delta = eval( str(self.beliefs['delta']).strip() )
        self.sigma = eval( str(self.beliefs['sigma']).strip() )
        
        # input scalings must be read if present
        if 'input_minmax' in self.beliefs:
            self.input_minmax=\
                eval( str(self.beliefs['input_minmax']).strip() )
        else:
            self.input_minmax=[]


        
    def final_beliefs(self, E, final=False):

        f="f" if final == True else ""
        n = str(E.tv_conf.no_of_trains)

        filename = E.config.beliefs + "-" + n + f

        print("New beliefs to file...")
        try:
            f=open(filename, 'w')
        except OSError as e:
            print("ERROR: Problem writing to file.")
            exit()

        f.write("active " + str(self.active) +"\n")
        f.write("output " + str(self.output) +"\n")
        f.write("basis_str "+ ' '.join(map(str,self.basis_str)) +"\n")
        f.write("basis_inf "+ "NA " + ' '.join(map(str,self.basis_inf)) +"\n")
        f.write("beta " + ' '.join(map(str,E.par.beta)) +"\n")
        f.write("fix_mean " + str(self.fix_mean) +"\n")
        f.write("kernel " + ' '.join(map(str,self.kernel))+"\n")
        f.write("delta " + str(E.par.delta) +"\n")
        input_minmax = [list(i) for i in E.all_data.minmax[:]]
        f.write("input_minmax "+ str(E.all_data.input_minmax) +"\n")
        f.write("sigma " + str(E.par.sigma) +"\n")
        f.close()


### bunch of hyperparameters stored here for convenience
class Hyperparams:
    def __init__(self, beliefs):
        self.beta = np.array(beliefs.beta)
        self.sigma = beliefs.sigma
        self.delta = beliefs.delta


### constructs the basis functions and stored them in the list 'h'
class Basis:
    def __init__(self, beliefs):
        self.h = []  # for list of basis functions

        # since stored inputs are only the 'active' inputs
        # we must readjust basis_inf to refer to the new indices
        # e.g. 0 1 3 -> 0 1 2 (the 3rd input is now, really, the 2nd)
        j=0
        if beliefs.active != []:
            for i in range(0, len(beliefs.basis_inf) ):
                if beliefs.basis_inf[i-j] not in beliefs.active:
                    print("Input", beliefs.basis_inf[i-j], "not active")
                    del beliefs.basis_inf[i-j]    
                    del beliefs.basis_str[i+1-j]
                    j=j+1
        beliefs.basis_inf = list( range(0,len(beliefs.basis_inf)) )

        self.make_h(beliefs.basis_str)
        self.print_mean_function\
          (beliefs.basis_inf, beliefs.basis_str, beliefs.active)
        self.basis_inf = beliefs.basis_inf

    # creates new basis function and appends to list
    def make_h(self, basis_str):
        comm = ""
        for i in range(0, len(basis_str) ):
            comm = comm\
              + "def h_" + str(i) + "(x):\n    return " + basis_str[i]\
              + "\nself.h.append(h_" + str(i) + ")\n"
        exec(comm)

    def print_mean_function(self, basis_inf, basis_str, include):
        self.meanf = "m(x) ="
        for i in range(0,len(self.h)):
            if i == 0 :
                self.meanf = self.meanf + " b"
            if i > 0 :
                if include == []:
                    self.meanf = self.meanf + " +" + " b"\
                      + str(basis_inf[i-1])\
                      + basis_str[i]\
                      + "["+str(basis_inf[i-1]) +"]"
                else:
                    self.meanf = self.meanf + " +" + " b"\
                      + str(include[i-1])\
                      + basis_str[i]\
                      + "["+str(include[i-1]) +"]"
        print(self.meanf)


### the configuration settings for training and validation
class TV_config:
    def __init__(self,k,c,noV):
        self.k=k  # number of sets to split data into
        self.c=c  # which validation set to use first (currently redundant)
        self.noV=noV  # how many validation sets
        self.retrain='y'
        self.no_of_trains=0
        self.auto=False
    
    def auto_train(self, auto):
        self.auto = True if auto else False

    def next_train(self):
        self.no_of_trains = self.no_of_trains+1

    def next_Vset(self):
        self.c=self.c+1

    def check_still_training(self):
        if self.no_of_trains < self.noV:
            if self.auto!=True and self.no_of_trains>=1:
                self.retrain = input("Retrain with V in T against new V? y/[n]: ")
            else:
                self.retrain='y'
        else:
            self.retrain='n'

        return True if self.retrain == 'y' else False

    def doing_training(self):
        if self.no_of_trains < self.noV:
            if self.retrain == 'y':
                self.next_train()
                return True
            else:
                return False
        else:
            self.retrain=='n'
        
    def do_final_build(self):
        if self.auto!=True:
            self.retrain=input("\nRetrain with V in T? y/[n]: ")
        else:
            self.retrain='y'

        return True if self.retrain == 'y' else False


### all data stores all requested data and splits it into T and V
class All_Data:
    def __init__(self, all_inputs, all_outputs, tv, beliefs, par,\
                 datashuffle, scaleinputs):

        print("\n*** Reading data files ***")

        print("Reading inputs file:", all_inputs)
        try:
            self.x_full = np.loadtxt(all_inputs)
        except OSError as e:
            print("ERROR: Problem reading file.")
            exit()
                
        print("Reading outputs file:", all_outputs)
        try:
            self.y_full=(np.loadtxt(all_outputs,usecols=[beliefs.output])).T
            print("Using output dimension",beliefs.output)
        except OSError as e:
            print("ERROR: Problem reading file.")
            exit()
     

        self.dim = self.x_full[0].size

        # if 1D inputs, store in 2D array with only 1 column
        if self.dim == 1:
            self.x_full = np.array([self.x_full,]).T

        ## option for which inputs to include
        if beliefs.active != []:
            print("Including input dimensions",beliefs.active)
            self.x_full = self.x_full[:,beliefs.active]



        self.input_minmax = beliefs.input_minmax
        self.map_inputs_0to1(par, scaleinputs)


        self.T=0
        self.V=0

        self.data_shuffle(datashuffle)

        self.tv=tv
        self.split_T_V_config()


    ## uses actual min and max of inputs
    def map_inputs_0to1(self, par, scaleinputs):
        minmax_l = []
        if scaleinputs == False:
            print("Input scaling off.")
            for i in range(0,self.x_full[0].size):
                templist = ( 0.0, 1.0 )
                minmax_l.append(templist)
            self.minmax = np.array(minmax_l)
        else:
            if self.input_minmax == []:
                print("Input scaling based on data.")
                for i in range(0,self.x_full[0].size):
                    templist =\
                      ( np.amin(self.x_full[:,i]) , np.amax(self.x_full[:,i]) )
                    minmax_l.append(templist)
                self.minmax = np.array(minmax_l)
            else:
                print("Input scaling based on \"input_minmax\" in beliefs file.")
                self.minmax = np.array(self.input_minmax)
        for i in range(0,self.x_full[0].size):
            self.x_full[:,i] = (self.x_full[:,i]-self.minmax[i,0])\
                             / (self.minmax[i,1]-self.minmax[i,0])
            #print("Dim",i,"scaled by %",(self.minmax[i,1]-self.minmax[i,0]))

 
    def data_shuffle(self, datashuffle):
        if datashuffle:
            print("Shuffling", self.x_full[:,0].size , "data points") 
            z_full = np.zeros\
              ( (self.x_full[:,0].size,self.x_full[0].size+self.y_full[0].size) )

            for i in range(0, self.x_full[0].size):
                z_full[:,i] = self.x_full[:,i]
            z_full[:,self.x_full[0].size] = self.y_full

            np.random.shuffle(z_full)

            for i in range(0, self.x_full[0].size):
                self.x_full[:,i] = z_full[:,i] 
            self.y_full = z_full[:,self.x_full[0].size]
        else:
            print("Data shuffling turned off.")


    def split_T_V_config(self):
        # k must be a factor of n
        if self.x_full[:,0].size % self.tv.k != 0:
            print("WARNING: 1st tv_conf value should be factor "
                  "of the number of data points.")
            exit()
        print("Split data into", self.tv.k, "sets")
        # c is the subset no. of the full data to use as V
        self.T=int((self.x_full[:,0].size/self.tv.k)*(self.tv.k-self.tv.noV))
        # V is the size of a single validation set
        self.V=int((self.x_full[:,0].size/self.tv.k)*(1))
        print("T-points:",self.T,"V-points:",self.V, "no. of V sets:",self.tv.noV)


    def choose_T(self):
        T_list = []
        V_list_all = []
        T_list = T_list + list(range(0,self.tv.c*self.V)) + list(range((self.tv.c+self.tv.noV)*self.V,self.tv.k*self.V))
        ## contains ALL the potential validation points
        V_list_all = V_list_all + list(range((self.tv.c)*self.V,(self.tv.c+self.tv.noV)*self.V))

        x_train=self.x_full[T_list,:]
        y_train=self.y_full[T_list]

        return (x_train, y_train)

    def choose_V(self):
        #print("Using set",self.tv.c,"as initial V")
        V_list = []
        V_list = V_list + list(range(self.tv.c*self.V,(self.tv.c+1)*self.V))
        x_valid=self.x_full[V_list,:]
        y_valid=self.y_full[V_list]
        return (x_valid, y_valid)

    def choose_new_V(self,validation):
        #print("***Using set",self.tv.c,"as new V***")
        V_list = []
        V_list = V_list + list(range(self.tv.c*self.V,(self.tv.c+1)*self.V))
        x_valid=self.x_full[V_list,:]
        y_valid=self.y_full[V_list]
        validation.inputs=x_valid
        validation.outputs=y_valid


### class for Data (training and validation) and associated structures
class Data:
    def __init__(self, inputs, outputs, basis, par, beliefs, K):
        self.inputs = inputs
        self.outputs = outputs
        self.basis = basis
        self.beliefs = beliefs
        self.par = par
        self.H = np.zeros([self.inputs[:,0].size, len(self.basis.h)])
        self.make_H()
        self.K = K
        self.make_A()

    # remake matrices
    def remake(self):
        self.make_H()
        self.make_A()

    # create H = (h(x1), h(x2) ...)
    def make_H(self):
        for i in range(0, self.inputs[:,0].size):
            for j in range(0, len(self.basis.h)):
                if j==0:
                    # first basis func returns 1.0
                    self.H[i,j]=self.basis.h[j](1.0)
                if j>0:
                    self.H[i,j]=self.basis.h[j]\
                      (self.inputs[i, self.basis.basis_inf[j-1]])
        
    # create Estimate
    def make_E(self):
        self.E = (self.H).dot(self.par.beta)

    def make_A(self):
        self.A = self.K.run_var_list(self.inputs)


### posterior distrubution, and also some validation tests
class Posterior:
    def __init__(self, Dnew, Dold, par, beliefs, K):
        self.Dnew = Dnew
        self.Dold = Dold
        self.par = par
        self.beliefs = beliefs
        self.K = K
        self.make_covar()
        self.new_new_mean()
        self.new_new_var()
        self.interval()

    def remake(self):
        self.make_covar()
        self.new_new_mean()
        self.new_new_var()
        self.interval()

    def make_covar(self):
        self.covar = self.K.run_covar_list(self.Dold.inputs, self.Dnew.inputs)

    def new_new_mean(self):
        self.newnewmean = self.Dnew.H.dot( self.par.beta ) + np.transpose(self.covar).dot(\
            linalg.solve( self.Dold.A, (self.Dold.outputs-self.Dold.H.dot(self.par.beta)) )\
        )

    def new_new_var_sub1(self):
        return self.Dnew.H - np.transpose(self.covar).dot( linalg.solve( self.Dold.A , self.Dold.H ) )

    def new_new_var_sub2(self):
        return np.transpose(self.Dold.H).dot( linalg.solve( self.Dold.A , self.Dold.H ) )

    def new_new_var_sub3(self):
        return np.transpose( self.covar ).dot( linalg.solve( self.Dold.A , self.covar ) )

    def new_new_var(self):
        self.newnewvar =\
          self.Dnew.A - self.new_new_var_sub3()\
        + self.new_new_var_sub1().dot( linalg.solve( self.new_new_var_sub2() , np.transpose(self.new_new_var_sub1()) ) )

    # return vectors of lower and upper 95% confidence intervals
    def interval(self):
        self.LI , self.UI = np.zeros([self.newnewmean.size]), np.zeros([self.newnewmean.size])
        for i in range(0, self.newnewmean.size):
            self.LI[i] = self.newnewmean[i] - 1.96 * np.sqrt( np.abs( self.newnewvar[i,i] ) )
            self.UI[i] = self.newnewmean[i] + 1.96 * np.sqrt( np.abs( self.newnewvar[i,i] ))


    def indiv_standard_error(self, ise):
        ## ise is the tolerance - 2.0 is good
        retrain=False
        e = np.zeros(self.Dnew.inputs[:,0].size)
        for i in range(0,e.size):
            e[i] = (self.Dnew.outputs[i]-self.newnewmean[i])\
                    /np.sqrt(self.newnewvar[i,i])
            if np.abs(e[i])>=ise:
                print("  Bad predictions:",self.Dnew.inputs[i,:],"ise:",np.round(e[i],decimals=4))
                retrain=True
        return retrain

    def mahalanobis_distance(self):
        ## ise is the tolerance
        retrain=False

        MDtheo = self.Dnew.outputs.size
        MDtheovar = 2*self.Dnew.outputs.size*\
            (self.Dnew.outputs.size+self.Dold.outputs.size\
                -self.par.beta.size-2.0)/\
            (self.Dold.outputs.size-self.par.beta.size-4.0)
        print("theoretical mahalanobis_distance (mean, var): (", MDtheo, "," , MDtheovar, ")")

        MD = ( (self.Dnew.outputs-self.newnewmean).T ).dot\
             ( linalg.solve( self.newnewvar , (self.Dnew.outputs-self.newnewmean) ) )
        print("calculated mahalanobis_distance:", MD)
        retrain=True
        return retrain

    def incVinT(self):
        print("Include V into T")
        self.Dold.inputs=np.append(self.Dnew.inputs,self.Dold.inputs,axis=0)
        self.Dold.outputs=np.append(self.Dnew.outputs,self.Dold.outputs)
        print("No. Training points:" , self.Dold.inputs[:,0].size)
       
        self.Dold.H=np.zeros([self.Dold.inputs[:,0].size,len(self.Dold.basis.h)])
        self.Dold.A=np.zeros([self.Dold.inputs[:,0].size,self.Dold.inputs[:,0].size])


    def final_design_points(self, E, final=False):

        f="f" if final == True else ""
        n = str(E.tv_conf.no_of_trains)
        o = str(E.beliefs.output)
        i_file  = E.config.inputs  + "-o" + o + "-" + n + f
        o_file  = E.config.outputs + "-o" + o + "-" + n + f

        # unscale the saved inputs before saving
        data4file = np.copy(self.Dold.inputs)
        for i in range(0,data4file[0].size):
            data4file[:,i] = data4file[:,i]\
              *(E.all_data.minmax[i,1]-E.all_data.minmax[i,0])+E.all_data.minmax[i,0]

        print("Writing T-data to:", i_file)
        try:
            np.savetxt(i_file, data4file, delimiter=' ', fmt='%.8f')
        except OSError as e:
            print("ERROR: Problem writing to file.")
            exit()
        
        print("Writing T-data to:", o_file)
        try:
            np.savetxt(o_file, self.Dold.outputs, delimiter=' ', fmt='%.8f')
        except OSError as e:
            print("ERROR: Problem writing to file.")
            exit()
