import numpy as np
import matplotlib.pyplot as plt
import sys

#--------------------------------------------------------------------------
# HMM Class


class HMM:
    ''' Paramters:
        N -- number of states
        M -- number of observation symbols
        T -- length of observation sequence '''
    def __init__(self, f_name):
        # Initialize HMM parameters
        with open(f_name, 'r') as f:
            _NMT = f.readline().strip().split(' ')
            self.N, self.M, self.T = [int(s) for s in _NMT]
            
            _states = f.readline().strip().split(' ')
            _vocab = f.readline().strip().split(' ')

            self.stateName = _states
            # Convert to dictionaries
            self.states = {state:i for i, state in enumerate(_states)}
            self.vocab = {obs:i for i, obs in enumerate(_vocab)}
            
            self.A, self.B, self.Pi = [], [], []

            # A matrix
            f.readline() # holds a:\r\n    
            line = f.readline().strip()
            while line != 'b:':
                self.A.append([float(a) for a in line.split(' ')])
                line = f.readline().strip()

            # B matrix
            line = f.readline().strip()
            while line != 'pi:':
                self.B.append([float(b) for b in line.split(' ')])
                line = f.readline().strip()

            # Pi matrix
            line = f.readline().strip()
            self.Pi.extend([float(pi) for pi in line.split(' ')])

            # Convert to numpy arrays
            self.A = np.array(self.A)
            self.B = np.array(self.B)
            self.Pi = np.array(self.Pi)

    def recognize(self, O):        
        # Base case
        v = self.vocab[O[0]] # index of observation symbol

        _alpha = self.Pi * self.B[:, v] # alpha at time t - 1
        alpha = np.zeros(self.N) # current alpha at time t
        
        # Induction
        for t in range(1, len(O)):
            v = self.vocab[O[t]] # update current observation

            for j in range(self.N):
                alpha[j] = np.sum(_alpha * self.A[:, j]) * self.B[j, v]
            _alpha = alpha.copy()
        
        # Termination
        return np.sum(alpha)
        
    
#--------------------------------------------------------------------------
#### read .obs file ###
# line 1 - number of datasets
# line 2 - number of observation + observations
#--------------------------------------------------------------------------
def init_observations(f_name):
    dataset = []
    
    f = open(f_name, 'r')
    num_obs = int(f.readline().strip())
    
    for seq in range(num_obs):
        seq_len = int(f.readline().strip())
        sequence = f.readline().strip().split(' ')
        dataset.append(sequence)
    f.close()
    return dataset
    

# O - Observations
# S - states
# A - State Transition Matrix
# B - Emission Matrix - Observation Probability Matrix
# PI - inintial state distance

def viterbi( obs, model ):
    
    states = model.states
    PI = model.Pi
    transA_p = model.A
    emitB_p = model.B
    
    V = [{}]
                
    ### --- Initialization --- ###
    for i, val in enumerate(states):
        #V[0][i] = PI[i]*emitB_p[i][obs[0]]
        idx = model.vocab[obs[0]]
        V[0][i] = PI[i]*emitB_p[i][idx]

    ### --- Recursion --- ###
    for t in range(1, len(obs)):
        V.append({})
        for y in states:
            idx = model.vocab[obs[t]]
            yidx = model.states[y]
            #prob = max(V[t - 1][y0]*transA_p[y0][y]*emitB_p[y][idx] for y0 in states)
            prob = []
            for y0 in states:
                y0idx = model.states[y0]
                temp = V[t - 1][y0idx]*transA_p[y0idx][yidx]*emitB_p[yidx][idx]
                prob.append(temp)
            
            V[t][yidx] = max(prob)

    opt = []
    for j in V:
        for x, y in j.items():
            if (j[x] == max(j.values())):
                opt.append(x)
    
    #Termination
    # The highest probability
    h = max(V[-1].values())
        
    output = []
    output.append(str(h))
    
    if h > 0.0:    
        for i in opt:
            output.append(model.stateName[i])

    print (' '.join(output) )
 
def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%10d" % i) for i in range(len(V)))
    for y in V[0]:
        yield "%.7s: " % y+" ".join("%.7s" % ("%f" % v[y]) for v in V)


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

if __name__ == '__main__': 
    model = HMM(sys.argv[1])
    dataset = init_observations(sys.argv[2])
        
    probs = []

    for O in dataset:
        viterbi(O, model )
    




