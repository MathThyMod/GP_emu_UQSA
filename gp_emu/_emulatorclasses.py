from __future__ import print_function
from builtins import input
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


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
        self.config = {}
        self.read_file()

    def read_file(self):
        print("*** Reading config file:", self.config_file ,"***")
        with open(self.config_file, 'r') as f:
            for line in f:
                (key, val) = line.split(' ',1)
                self.config[key] = val

        self.beliefs = str(self.config['beliefs']).strip()
        self.inputs = str(self.config['inputs']).strip()
        self.outputs = str(self.config['outputs']).strip()

        self.tv_config = tuple(str(self.config['tv_config']).strip().split(' '))
        self.tv_config =\
            [int(self.tv_config[i]) for i in range(0, len(self.tv_config))]
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
                print("constraints must be T or F")
        print("constraints:" , self.constraints)

        stochastic = str(self.config['stochastic']).strip()
        if stochastic == 'T':
            self.stochastic = True
        else:
            if stochastic == 'F':
                self.stochastic = False
            else:
                print("stochastic must be T or F")
        print("stochastic:" , self.stochastic)

        self.constraints_type = str(self.config['constraints_type']).strip()
        print("constraints_type:", self.constraints_type)


### gathers all the beliefs from the specified belief file
class Beliefs:
    def __init__(self,beliefs_file):
        self.beliefs_file=beliefs_file
        self.beliefs = {}
        self.read_file()
        
    def read_file(self):
        ## read into a dictionary
        print("\n*** Reading beliefs file:" , self.beliefs_file , "***")
        with open(self.beliefs_file, 'r') as f:
            for line in f:
                (key, val) = line.split(' ',1)
                self.beliefs[key] = val

        self.basis_str = (str(self.beliefs['basis_str']).strip().split(' '))
        self.basis_inf =\
          [int(i) for i in str(self.beliefs['basis_inf']).strip().split(' ')[1:]]
        self.beta =\
          [float(i) for i in (str(self.beliefs['beta']).strip().split(' '))]
        self.fix_mean = str(self.beliefs['fix_mean']).strip().split(' ')[0]
        kernel_list = str(self.beliefs['kernel']).strip().split(' ')
        self.kernel = kernel_list
        delta_str = str(self.beliefs['delta']).strip()
        self.delta = eval( delta_str )
        sigma_str = str(self.beliefs['sigma']).strip()
        self.sigma = eval( sigma_str )
        
        # input scalings must be read in if present
        if 'input_minmax' in self.beliefs:
            self.input_minmax=\
                eval( str(self.beliefs['input_minmax']).strip() )
        else:
            self.input_minmax=[]

        self.active = str(self.beliefs['active']).strip().split(' ')[0:]
        if self.active[0] == "all" or self.active[0] == "[]":
            self.active = []
        else:
            self.active= list(self.active)
            self.active=[int(self.active[i]) for i in range(0, len(self.active))]
        print("active:", self.active)
        
        self.output = int( str(self.beliefs['output']).strip().split(' ')[0] )
        print("output:",self.output)

        
    def final_beliefs(self, E, final=False):

        config = E.config
        par = E.par
        minmax = E.all_data.minmax
        K = E.K

        f="f" if final == True else ""
        n = str(E.tv_conf.no_of_trains)
        o = str(E.beliefs.output)

        filename = config.beliefs + "-" + n + f

        print("New beliefs to file...")
        f=open(filename, 'w')
        f.write("active " + str(self.active) +"\n")
        f.write("output " + str(self.output) +"\n")
        f.write("basis_str "+ ' '.join(map(str,self.basis_str)) +"\n")
        f.write("basis_inf "+ "NA " + ' '.join(map(str,self.basis_inf)) +"\n")
        f.write("beta " + ' '.join(map(str,par.beta)) +"\n")
        f.write("fix_mean " + str(self.fix_mean) +"\n")
        f.write("kernel " + ' '.join(map(str,self.kernel))+"\n")
        f.write("delta " + str(par.delta) +"\n")
        input_minmax = [list(i) for i in minmax[:]]
        f.write("input_minmax "+ str(input_minmax) +"\n")
        f.write("sigma " + str(par.sigma) +"\n")
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
        self.h = []
 
        j=0 ## j corrects for the shortening of the basis lists
        if beliefs.active != []:
            for i in range(0, len(beliefs.basis_inf) ):
                if beliefs.basis_inf[i-j] not in beliefs.active:
                    print("Input", beliefs.basis_inf[i-j], "not active")
                    del beliefs.basis_inf[i-j]    
                    del beliefs.basis_str[i+1-j]
                    j=j+1
       
        ## basis_inf now has to refer to the correct new dimensions
        beliefs.basis_inf = list( range(0,len(beliefs.basis_inf)) )

        self.make_h(beliefs.basis_str)
        self.print_mean_function\
          (beliefs.basis_inf,beliefs.basis_str, beliefs.active)
        self.basis_inf=beliefs.basis_inf

    def make_h(self, basis_str):
        comm = ""
        for i in range(0, len(basis_str) ):
            comm = comm + "def h_" + str(i) + "(x):\n    return " + basis_str[i] + "\nself.h.append(h_" + str(i) + ")\n"
        exec(comm)# in locals()

    def print_mean_function(self,basinfo,basis_str, include):
        self.meanf="m(x) ="
        for i in range(0,len(self.h)):
            if i==0:
                self.meanf=self.meanf+" b"#+ str(i)
            if i>0:
                if include == []:
                    self.meanf=self.meanf+" +"+" b"+ str(basinfo[i-1]) + basis_str[i] + "["+str(basinfo[i-1]) +"]"
                else:
                    self.meanf=self.meanf+" +"+" b"+ str(include[i-1]) + basis_str[i] + "["+str(include[i-1]) +"]"
        print(self.meanf)

### the configuration settings for training and validation
class TV_config:
    def __init__(self,k,c,noV):
        self.k=k
        self.c=c
        self.noV=noV
        self.retrain='y'
        self.no_of_trains=0
        self.auto=False
    
    def auto_train(self, auto):
        if auto:
            self.auto = True
        else:
            self.auto = False

    def next_train(self):
        self.no_of_trains = self.no_of_trains+1
        print("\n***Training run no.:",self.no_of_trains,"***") 

    def next_Vset(self):
        self.c=self.c+1

    def check_still_training(self):
        if self.no_of_trains<self.noV:
            if self.auto!=True and self.no_of_trains>=1:
                self.retrain = input("Retrain against new V? y/[n]: ")
            else:
                self.retrain='y'
        else:
            self.retrain='n'
        if self.retrain=='y':
            return True
        else:
            return False

    def doing_training(self):
        if self.no_of_trains<self.noV:
            if self.retrain=='y':
                self.next_train()
                return True
            else:
                return False
        else:
            self.retrain=='n'
        
    def do_final_build(self):
        if self.auto!=True:
            self.retrain=input("\nRetrain with V before full predict? y/[n]: ")
        else:
            self.retrain='y'
        if self.retrain=='y':
            return True
        else:
            return False


### all data stores all requested data and splits it into T and V
class All_Data:
    def __init__(self, all_inputs, all_outputs, tv, beliefs, par,\
                datashuffle, scaleinputs):

        print("\n*** Reading data files:",all_inputs,"&",all_outputs,"***")
        self.x_full=np.loadtxt(all_inputs)

        # if 1D inputs, store in 2D array with only 1 column
        if self.x_full[0].size==1:
            self.x_full = np.array([self.x_full,])
            self.x_full = self.x_full.transpose()

        self.dim = self.x_full[0].size

        ## option for which inputs to include
        self.include=[]
        if beliefs.active != []:
            print("Including input dimensions",beliefs.active)
            self.x_full = self.x_full[:,beliefs.active]

        self.datashuffle = datashuffle
        self.scaleinputs = scaleinputs

        self.input_minmax = beliefs.input_minmax

        self.map_inputs_0to1(par)

        print("Using output dimension",beliefs.output)
        self.y_full=(np.loadtxt(all_outputs,usecols=[beliefs.output])).transpose()

        self.T=0
        self.V=0
        ### I have removed the data shuffle for now
        if self.datashuffle == True:
            self.data_shuffle()
        else:
            print("Data shuffle turned off.")
        self.tv=tv
        self.split_T_V_config()
        
    ## uses actual min and max of inputs
    def map_inputs_0to1(self, par):
        minmax_l = []
        #print( "x_full:" , self.x_full )
        if self.scaleinputs == False:
            print("Input scaling turned off.")
            for i in range(0,self.x_full[0].size):
                templist = ( 0.0, 1.0 )
                minmax_l.append(templist)
            self.minmax = np.array(minmax_l)
        else:
            if self.input_minmax == []:
                for i in range(0,self.x_full[0].size):
                    templist=(np.amin(self.x_full[:,i]),np.amax(self.x_full[:,i]))
                    minmax_l.append(templist)
                self.minmax = np.array(minmax_l)
            else:
                self.minmax = np.array(self.input_minmax)
        for i in range(0,self.x_full[0].size):
            self.x_full[:,i] = (self.x_full[:,i]-self.minmax[i,0])/(self.minmax[i,1]-self.minmax[i,0])
            print("Dim",i,"scaled by %",(self.minmax[i,1]-self.minmax[i,0]))
   
        #print( "x_full:" , self.x_full )
 
    def data_shuffle(self):
        print("Random shuffle of",self.x_full[:,0].size,"input-output pairs") 
        z_full=np.zeros\
          ( (self.x_full[:,0].size,self.x_full[0].size+self.y_full[0].size) )

        for i in range(0,self.x_full[0].size):
            z_full[:,i]=self.x_full[:,i]
        z_full[:,self.x_full[0].size]=self.y_full

        np.random.shuffle(z_full)

        for i in range(0,self.x_full[0].size):
            self.x_full[:,i]=z_full[:,i] 
        self.y_full=z_full[:,self.x_full[0].size]


    def split_T_V_config(self):
        print("Split data into", self.tv.k,"sets")
        # k must be a factor of n
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
        #self.make_E()
        self.K = K
        self.make_A()

    # remake matrices
    def remake(self):
        self.make_H()
        #self.make_E()
        self.make_A()

    # create H = (h(x1), h(x2) ...)
    def make_H(self):
        for i in range(0, self.inputs[:,0].size):
            for j in range(0,len(self.basis.h)):
                if j==0:
                    self.H[i,j]=self.basis.h[j](1.0) # first basis func returns 1
                if j>0:
                    #print("basis_inf:" , self.basis.basis_inf[j-1])
                    self.H[i,j]=self.basis.h[j]\
                      (self.inputs[i,self.basis.basis_inf[j-1]])
        
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
            #linalg.inv(self.Dold.A).dot( self.Dold.outputs - self.Dold.H.dot(self.par.beta) )\
            linalg.solve( self.Dold.A, (self.Dold.outputs-self.Dold.H.dot(self.par.beta)) )\
        )

    def new_new_var_sub1(self):
        #return self.Dnew.H - np.transpose(self.covar).dot( linalg.inv(self.Dold.A) ).dot( self.Dold.H ) 
        return self.Dnew.H - np.transpose(self.covar).dot( linalg.solve( self.Dold.A , self.Dold.H ) )

    def new_new_var_sub2(self):
        #return np.transpose(self.Dold.H).dot( linalg.inv(self.Dold.A) ).dot( self.Dold.H )
        return np.transpose(self.Dold.H).dot( linalg.solve( self.Dold.A , self.Dold.H ) )

    def new_new_var_sub3(self):
        #return np.transpose( self.covar ).dot( linalg.inv(self.Dold.A) ).dot( self.covar ) 
        return np.transpose( self.covar ).dot( linalg.solve( self.Dold.A , self.covar ) )

    def new_new_var(self):
        self.newnewvar =\
          self.Dnew.A - self.new_new_var_sub3()\
        #+ self.new_new_var_sub1().dot( linalg.inv(self.new_new_var_sub2()) ).dot( np.transpose(self.new_new_var_sub1()) )
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

        print("Design points to I/O files:\n", i_file , "&" , o_file)
        np.savetxt(i_file, data4file, delimiter=' ', fmt='%.8f')
        np.savetxt(o_file, self.Dold.outputs, delimiter=' ', fmt='%.8f')
