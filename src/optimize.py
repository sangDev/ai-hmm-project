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

            self.stateName = _states
            self.vocabName = _vocab            
            
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

    def __str__(self):
        NMT = '{} {} {}'.format(self.N, self.M, self.T)
        states = ' '.join(self.stateName)
        vocab = ' '.join(self.vocabName)
        A = '\n'.join([' '.join(['{:.6f}'.format(cell) for cell in row]) for row in self.A])
        B = '\n'.join([' '.join(['{:.6f}'.format(cell) for cell in row]) for row in self.B])
        Pi = ' '.join(['{:.6f}'.format(pi) for pi in self.Pi])

        str_HMM = [NMT, states, vocab, A, B, Pi]
        return '{}\n{}\n{}\na:\n{}\nb:\n{}\npi:\n{}\n'.format(*str_HMM)
    
    def _alpha(self, O):
        alpha = np.zeros((len(O), self.N))
        # Base case
        v = self.vocab[O[0]] # index of observation symbol
        alpha[0] = self.Pi * self.B[:, v] # alpha at time 0
        
        # Induction
        for t in range(1, len(O)):
            v = self.vocab[O[t]] # update current observation

            for j in range(self.N):
                alpha[t][j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, v]

        return alpha

    def _beta(self, O):
        # Initialization
        beta = np.zeros((len(O), self.N))
        
        # Base case
        beta[-1] = np.ones(self.N)

        # Induction
        for t in range(len(O) - 1, 0, -1):
            v = self.vocab[O[t]] # update current observation

            for i in range(self.N):
                beta[t-1][i] = np.sum(self.A[i] * self.B[:, v] * beta[t])

        return beta
    
    def _gamma(self, O):
        gamma = np.zeros((len(O), self.N))
        alpha = self._alpha(O)
        beta = self._beta(O)

        gamma = alpha * beta
        _sum = np.sum(gamma, axis=1)
        _sum.shape = (_sum.shape[0], 1)
        for i, v in enumerate(_sum):
            if _sum[i] != 0:
                gamma[i] /= v

        return gamma
        
    def get_xi_bar(self, O):
        ''' Iteratively computes the sum of the xi variable in the
            Baum-Welch estimation algorithm from 1 to len(O) - 1'''
        # We could calculate the entire 3D array then sum the values, but
        # for space considerations lets collapse it into a 2D array and
        # add up the time values iteratively.
        alpha = self._alpha(O)
        beta = self._beta(O)
        
        xi_bar_sum = np.zeros((self.N, self.N))
        
        for t in range(len(O) - 1):
            v = self.vocab[O[t+1]] # Output at time t+1
            xi_bar = np.zeros((self.N, self.N))            
            for i in range(self.N):
                for j in range(self.N):
                    xi_bar[i,j] += alpha[t,i] * self.A[i,j] * self.B[j, v] * beta[t+1,j]

            _sum = xi_bar.sum()
            if _sum == 0:
                break
            xi_bar /= _sum
            
            xi_bar_sum += xi_bar            
        
        return xi_bar_sum

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

    def optimize(self, O):
        gamma = self._gamma(O)
        xi_bar_sum = self.get_xi_bar(O)
        
        self.Pi = gamma[0]
      
        sum_gamma = gamma[:-1].sum(axis=0)
        
        for i in range(len(sum_gamma)):
            if (sum_gamma[i] == 0.0):
                self.A[i][:] = self.A[i][:]
            else:
                self.A[i][:] = xi_bar_sum[i][:] / sum_gamma[i]
                
        # denominator of beta_bar calculation
        sum_gamma_toT = gamma.sum(axis=0)       
        
        # get gamma such that Ot = Vk

        sum_gamma_OtVk = np.zeros((self.N, self.M))        
        
        for j in range(self.N):
            for k in range (self.M):
                for t in range(len(O)):
                    #print t
                    if (O[t] == self.vocabName[k]):               
                        sum_gamma_OtVk[j][k] += gamma[t][j]
                    else:
                        sum_gamma_OtVk[j][k] += 0
 
         # for each gamma to T check if its divide by zero
        for i in range(len(sum_gamma_toT)):
            if (sum_gamma_toT[i] == 0.0):
                self.B[i][:] = self.B[i][:]
            else:
                self.B[i][:] = sum_gamma_OtVk[i][:] / sum_gamma_toT[i]

        
if __name__ == '__main__':
    model = HMM(sys.argv[1])
    dataset = init_observations(sys.argv[2])
    model.T = len(max(dataset, key=lambda x:len(x)))

    new_hmm = sys.argv[3]

    for O in dataset:
        old_p = model.recognize(O)
        model.optimize(O)
        new_p = model.recognize(O)

        print '{:.6f} {:.6f}'.format(old_p, new_p)
    
    with open(new_hmm, 'w') as f:
        f.write(str(model))
