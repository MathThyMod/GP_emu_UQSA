from __future__ import print_function
from builtins import input
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
#import emulatorkernels as emuk

### gathers everything together for convenience
class Emulator:
    def __init__(self,beliefs,par,basis,tv_conf,all_data,training,validation,post,opt_T, K):
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
    def __init__(self,optconfig_file):
        self.config_file=optconfig_file
        self.config = {}
        self.read_file()

    def read_file(self):
        print("\n***Reading config file...***")
        with open(self.config_file, 'r') as f:
            for line in f:
                (key, val) = line.split(' ',1)
                self.config[key] = val

        self.beliefs = str(self.config['beliefs']).strip()
        print(self.beliefs)
        self.inputs = str(self.config['inputs']).strip()
        print(self.inputs)
        self.outputs = str(self.config['outputs']).strip()
        print(self.outputs)

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
        print("\n***Reading beliefs file...***")
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
#        print(kernel_list)
#        kernel_list =\
#           list( str(self.beliefs['kernel']).strip() )
        self.kernel = kernel_list
        #print("kernel:" , self.kernel)
        delta_str = str(self.beliefs['delta']).strip()
        self.delta = eval( delta_str )
        sigma_str = str(self.beliefs['sigma']).strip()
        self.sigma = eval( sigma_str )
        #self.nugget = float(str(self.beliefs['nugget']).strip().split(' ')[0])
        #print("nugget:", self.nugget)
        
        if 'input_minmax' in self.beliefs:
            ## input scalings must be read in if present
            self.input_minmax=\
                eval( str(self.beliefs['input_minmax']).strip() )
        else:
            self.input_minmax=[]

        #print("*** from beliefs ***" , self.input_minmax)

        self.active = str(self.beliefs['active']).strip().split(' ')[0:]
        if self.active[0] == "all" or self.active[0] == "[]":
            self.active = []
        else:
            self.active= list(self.active)
            self.active=[int(self.active[i]) for i in range(0, len(self.active))]
        print("active:", self.active)
        
        self.output = int( str(self.beliefs['output']).strip().split(' ')[0] )
#        self.output= list(self.output)
#        self.output=[int(self.output[i]) for i in range(1, len(self.output))]
        print("output:",self.output)
        
    def final_beliefs(self, filename, par, minmax, K):
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
        #print("******* " , input_minmax)
        f.write("input_minmax "+ str(input_minmax) +"\n")
        f.write("sigma " + str(par.sigma) +"\n")
        #f.write("nugget " + str(K.nugget) +"\n")
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
#        if auto == True:
#            self.auto = True
#        else:
#            #if input("\nSet to auto train? y/[n]: ") == 'y':
#            self.auto = False
    
    def auto_train(self):
        self.auto=True

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
        if self.retrain=='y':
            self.next_train()
            return True
        else:
            return False
        
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
        print("\n***Data from",all_inputs,all_outputs,"***")
        self.x_full=np.loadtxt(all_inputs)
        if self.x_full[0].size==1:
            print("GP_emu doesn't support 1D inputs, sorry! Exiting...")
            exit()
            self.x_full = np.array([self.x_full,])
            self.x_full = self.x_full.transpose()
            print("1D data in 2D array, shape:",self.x_full.shape)

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
        self.y_full=(np.loadtxt(all_outputs, usecols=[beliefs.output])).transpose()

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
        #print("Split data into", self.tv.k,"sets")
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
        #self.Dold.inputs=np.append(self.Dold.inputs,self.Dnew.inputs,axis=0)
        self.Dold.inputs=np.append(self.Dnew.inputs,self.Dold.inputs,axis=0)
        #self.Dold.outputs=np.append(self.Dold.outputs,self.Dnew.outputs)
        self.Dold.outputs=np.append(self.Dnew.outputs,self.Dold.outputs)
        print("No. Training points:" , self.Dold.inputs[:,0].size)
        
        self.Dold.H=np.zeros([self.Dold.inputs[:,0].size,len(self.Dold.basis.h)])
        self.Dold.A=np.zeros([self.Dold.inputs[:,0].size,self.Dold.inputs[:,0].size])


    def final_design_points(self, final_design_file, final_design_file_o, minmax):
        data4file = np.copy(self.Dold.inputs)
        for i in range(0,data4file[0].size):
            data4file[:,i] = data4file[:,i]*(minmax[i,1]-minmax[i,0]) + minmax[i,0]

        print("Final design points to I/O files:",\
               final_design_file,"&",final_design_file_o,"...")
        #np.savetxt(final_design_file, data4file.transpose(),\
        #  delimiter=' ', fmt='%1.4f')
        #np.savetxt(final_design_file_o, self.Dold.outputs.transpose(),\
        #  delimiter=' ', fmt='%1.4f')
        np.savetxt(final_design_file, data4file,\
          delimiter=' ', fmt='%1.4f')
        np.savetxt(final_design_file_o, self.Dold.outputs,\
          delimiter=' ', fmt='%1.4f')
         
           
### for optimising the hyperparameters
class Optimize:
    def __init__(self, data, basis, par, beliefs, config):
        self.data = data
        self.basis = basis
        self.par = par
        self.beliefs = beliefs
        self.standard_constraint()
        
        # if bounds are empty then construct them automatically
        if config.bounds == ():
            bounds_t = []
            for d in range(0, len(self.data.K.delta)):
                for i in range(0,self.data.K.delta[d].size):
                    bounds_t.append([0.001,1.0])
            for s in range(0, len(self.data.K.delta)):
                for i in range(0,self.data.K.sigma[s].size):
                    bounds_t.append([0.001,100.0])
            config.bounds = tuple(bounds_t)
            print("bounds not provided, so setting to:")
            print(config.bounds)

        # set up type of bounds
        if config.constraints_type == "bounds":
            self.bounds_constraint(config.bounds)
        else:
            if config.constraints_type == "noise":
                self.noise_constraint()
            else:
                print("Constraints set to standard")
                self.standard_constraint()
        
    ## tries to keep deltas above a small value
    def standard_constraint(self):
        self.cons=[]
        for i in range(0,self.data.K.delta_num):
            hess = np.zeros(self.data.K.delta_num+self.data.K.sigma_num)
            hess[i]=1.0
            dict_entry= {\
                        'type': 'ineq',\
                        'fun' : lambda x, f=i: x[f]-2.0*np.log(0.001),\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
        self.cons = tuple(self.cons)
        

    ## tries to keep within the bounds as specified for global stochastic opt
    def bounds_constraint(self, bounds):
        print("Setting full bounds constraint")
        self.cons=[]
        for i in range(0,self.data.K.delta_num):
            hess = np.zeros(self.data.K.delta_num+self.data.K.sigma_num)
            hess[i]=1.0
            lower, upper = bounds[i]
            dict_entry= {\
                        'type': 'ineq',\
                'fun' : lambda x, a=lower, f=i: x[f]-2.0*np.log(a),\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
            dict_entry= {\
                        'type': 'ineq',\
                'fun' : lambda x, b=upper, f=i: 2.0*np.log(b) - x[f],\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
        self.cons = tuple(self.cons)


    ## tries to keep deltas above a small value, and fixes sigma noise
    def noise_constraint(self):
        ## assume last sig is noise
        if self.data.K.name[-1] == "Noise" :
            n = self.data.K.delta_num+self.data.K.sigma_num
            noise_val = 2.0*np.log(self.data.K.sigma[-1][0])
            self.cons=[]
            hess = np.zeros(n)
            hess[n-1]=1.0
            dict_entry= {\
                        'type': 'ineq',\
                'fun' : lambda x, f=n-1, nv=noise_val: x[f]-(nv-2.0*np.log(0.001)),\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
            dict_entry= {\
                        'type': 'ineq',\
                'fun' : lambda x, f=n-1, nv=noise_val: (nv+2.0*np.log(0.001))-x[f],\
                        'jac' : lambda x, h=hess: h\
                        }
            self.cons.append(dict_entry)
            for i in range(0,self.data.K.delta_num):
                hess = np.zeros(n)
                hess[i]=1.0
                dict_entry= {\
                            'type': 'ineq',\
                            'fun' : lambda x, f=i: x[f] - 2.0*np.log(0.001),\
                            'jac' : lambda x, h=hess: h\
                            }
                self.cons.append(dict_entry)
            self.cons = tuple(self.cons)
            print("Noise constraint set")
        else:
            print("Last kernel is not Noise, so Noise constraint won't work")


    def llhoptimize_full(self, numguesses, use_cons, bounds, stochastic, print_message=False):
        print("Optimising delta and sigma...")

        ### scale the provided bounds
        bounds_new = []
        for i in bounds:
            temp = 2.0*np.log(np.array(i))
            bounds_new = bounds_new + [list(temp)]
        bounds = tuple(bounds_new)

        self.optimal_full(numguesses, use_cons, bounds, stochastic, print_message)
        print("best delta: " , self.par.delta)
        print("best sigma: ", self.par.sigma)

        if self.beliefs.fix_mean=='F':
            self.optimalbeta()
        print("best beta: " , np.round(self.par.beta,decimals=4))

   
    def optimal_full(self, numguesses, use_cons, bounds, stochastic, print_message=False):
        first_try = True
        best_min = 10000000.0

        guessgrid =\
          np.zeros([self.data.K.delta_num + self.data.K.sigma_num, numguesses])

        ### construct list of guesses from bounds
        for R in range(0, self.data.K.delta_num + self.data.K.sigma_num): ## HERE
            BL = bounds[R][0]
            BU = bounds[R][1]
            guessgrid[R,:] = BL+(BU-BL)*np.random.random_sample(numguesses)

        ### try each x-guess (start value for optimisation)
        if stochastic:
            print("Using global stochastic method...")
        else:
            if use_cons:
                print("Using constrained COBYLA method...")
            else:
                print("Using Nelder-Mead method...")
        for C in range(0,numguesses):
            x_guess = list(guessgrid[:,C])
            if True:
                if stochastic:
                    #res = differential_evolution(self.loglikelihood_full, bounds)
                    while True: 
                        res = differential_evolution\
                          (self.loglikelihood_full, bounds, maxiter=200, tol=0.1)
                        if print_message:
                            print(res, "\n")
                        if res.success == True:
                            break
                        else:
                            print(res.message, "Trying again.")
                else:
                    while True: 
                        if use_cons:
                            res = minimize(self.loglikelihood_full,\
                              x_guess, constraints=self.cons, method = 'COBYLA',\
                              tol=0.1)
                            if print_message:
                                print(res, "\n")
                        else:
                            print("Using Nelder-Mead method...")
                            res = minimize(self.loglikelihood_full,
                              x_guess, method = 'Nelder-Mead',\
                              options={'xtol':0.1, 'ftol':0.001})
                            if print_message:
                                print(res, "\n")
                        if res.success == True:
                            break
                        else:
                            print(res.message, "Trying again.")
                print("  result: " , np.around(np.exp(res.x/2.0),decimals=4),\
                      " llh: ", -1.0*np.around(res.fun,decimals=4))
                #print("res.fun:" , res.fun)
                if (res.fun < best_min) or first_try:
                    best_min = res.fun
                    best_x = np.exp(res.x/2.0)
                    best_res = res
                    first_try = False
        print("********")
        self.x_to_delta_and_sigma(best_x)
        ## store these values in par, so we remember them
        self.par.delta = [[list(i) for i in d] for d in self.data.K.delta]
        self.par.sigma = [list(s) for s in self.data.K.sigma]

        self.data.make_A()
        self.data.make_H()
  
 
    def loglikelihood_full(self, x):
        #### undo the transformation...
        x = np.exp(x/2.0)
        self.x_to_delta_and_sigma(x)

        self.data.make_A()
        (signdetA, logdetA) = np.linalg.slogdet(self.data.A)
        #### slow
        ## invA = linalg.inv(self.data.A)
        ## val=linalg.det( ( np.transpose(self.data.H) ).dot( ( invA ).dot(self.data.H) ))

        #longexp =\
        #( np.transpose(self.data.outputs) )\
        #.dot(\
        # (\
        #   invA - ( invA.dot(self.data.H) )\
        #      .dot(\
        #         linalg.inv(\
        #            ( np.transpose(self.data.H) ).dot( ( invA ).dot( self.data.H ) )\
        #                      ) \
        #          )\
        #       .dot( np.transpose(self.data.H).dot( invA ) ) \
        # )\
        #    ).dot(self.data.outputs)


        #### fast - no direct inverses
        val=linalg.det( ( np.transpose(self.data.H) ).dot( linalg.solve( self.data.A , self.data.H )) )
        invA_f = linalg.solve(self.data.A , self.data.outputs)
        invA_H = linalg.solve(self.data.A , self.data.H)

        longexp =\
        ( np.transpose(self.data.outputs) )\
        .dot(\
           invA_f - ( invA_H ).dot\
              (\
                linalg.solve( np.transpose(self.data.H).dot(invA_H) , np.transpose(self.data.H) )
              )\
             .dot( invA_f )\
            )

        ## max the lll, i.e. min the -lll 
        #print(signdetA, val)
        ## we can get negative signdetA when problems with A e.g. ill-conditioned
        if signdetA > 0 and val > 0:
            return -(\
           -0.5*longexp - 0.5*(np.log(signdetA)+logdetA) - 0.5*np.log(val)\
           -0.5*(self.data.inputs[0].size - self.par.beta.size)*np.log(2.0*np.pi)\
                    )
        else:
            print("ill conditioned covariance matrix...")
            return 10000.0


    def optimalbeta(self):
        #### slow
        #self.par.beta = ( linalg.inv( np.transpose(self.data.H).dot( linalg.inv(self.data.A).dot( self.data.H ) ) ) ).dot\
    #( np.transpose(self.data.H).dot( linalg.inv(self.data.A) ).dot( self.data.outputs ) )

        #### fast - no direct inverses
        invA_f = linalg.solve(self.data.A , self.data.outputs)
        invA_H = linalg.solve(self.data.A , self.data.H)
        self.par.beta = linalg.solve( np.transpose(self.data.H).dot(invA_H) , np.transpose(self.data.H) ).dot(invA_f)


    def x_to_delta_and_sigma(self,x):
        x_read = 0
        x_temp = []
        for d in range(0, len(self.data.K.delta)):
            if self.data.K.delta[d].size > 0:
                d_per_dim = int(self.data.K.delta[d].flat[:].size/\
                  self.data.K.delta[d][0].size)
                x_temp.append(x[ x_read:x_read+self.data.K.delta[d].size ]\
                  .reshape(d_per_dim,self.data.K.delta[d][0].size))
            else:
                x_temp.append([])
            x_read = x_read + self.data.K.delta[d].size
        self.data.K.update_delta(x_temp)
 
        x_temp = []
        for s in range(0, len(self.data.K.sigma)):
            x_temp.append(x[ x_read:x_read+self.data.K.sigma[d].size ]) 
            x_read = x_read + self.data.K.sigma[s].size
        self.data.K.update_sigma(x_temp)
        return
 
