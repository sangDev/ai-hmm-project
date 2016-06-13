import sys
import numpy as np

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

if __name__ == '__main__': 
    model = HMM(sys.argv[1])
    dataset = init_observations(sys.argv[2])
    
    probs = []
    for O in dataset:
        pO = model.recognize(O)
        probs.append(str(pO))

    print '\n'.join(probs)
