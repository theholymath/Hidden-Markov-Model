import numpy as np
import time
import re

class HMM(object):
    """
    This HMM is fast and efficient and is based on the paper,
    "A Revealing Introduction to Hidden Markov Models."
    It inclused a method to parse the Brown Corpus (included in directory)
    The test cases at the bottom are from the paper cited here. 
    """
    def __init__(self):
        super(HMM, self).__init__()

    def get_stochastic_row_for_initialization(self,N):
        eps = (1./N)*(1/100.0)
        if N%2:
            row = [1./N + (-1.0)**i*eps for i in range(N-1)] + [1./N]
        else:
            row = [1./N + (-1.0)**i*eps for i in range(N)]

        return row

    def HMM(self,A,B,pi,Obs,maxIters=100):
        self.A   = A
        self.B   = B
        self.pi  = pi
        self.Obs = Obs
        self.maxIters = maxIters
        self.oldLogProb = -np.inf # may want to keep track
        self.iters = 0
        self._compute_params()

        while (self.iters < self.maxIters and self.newLogProb > self.oldLogProb):
            self.oldLogProb = self.newLogProb
            self._compute_params()
            self.iters+=1

        return self

    def _compute_params(self):
        N,M = self.B.shape
        T = len(self.Obs)

        c = [0.0]*T
        alpha  = np.ones((T,N))
        beta   = np.ones((T,N))
        gamma  = np.ones((T,N,N))
        gamma2 = np.ones((T,N))

        # compute a_0(i)
        alpha[0,:] = [self.pi[i]*self.B[i,int(self.Obs[0])] for i in range(N)]
        c[0] = np.sum(alpha[0,:])

        # scale a_0[i]
        c[0] = 1.0/c[0]
        alpha[0,:] = [c[0]*alpha[0,i] for i in range(N)]

        # compute a_t(i)
        for t in range(1,T):
            for i in range(N):
                alpha[t,i] = np.sum([alpha[t-1,j]*self.A[j,i] for j in range(N)])

                alpha[t,i] = alpha[t,i]*self.B[i,self.Obs[t]]
                c[t] += alpha[t,i]

            # scale alpha[t,i]
            c[t] = 1.0/c[t]

            alpha[t,:] = [c[t]*alpha[t,i] for i in range(N)]
        self.alpha = alpha

        #### BETA
        beta[T-1,:] = [c[T-1]]*N

        # beta pass
        for t in range(T-2,-1,-1):
            for i in range(N):
                beta[t,i] = np.sum([self.A[i,j]*self.B[j,self.Obs[t+1]]*beta[t+1,j] for j in range(N)])

                #scale by same factor
                beta[t,i] = c[t]*beta[t,i]

        # gamma
        for t in range(T-1):
            denom = 0.0
            for i in range(N):
                for j in range(N):
                    denom += alpha[t,i]*self.A[i,j]*self.B[j,self.Obs[t+1]]*beta[t+1,j]

            for i in range(N):
                gamma2[t,i] = 0.0
                for j in range(N):
                    gamma[t,i,j] = (alpha[t,i]*self.A[i,j]*self.B[j,self.Obs[t+1]]*beta[t+1,j])/denom
                    gamma2[t,i] += gamma[t,i,j]


        # special case
        denom = np.sum(alpha[T-1,:])
        gamma2[T-1,:] = [alpha[T-1,i]/denom for i in range(N)]
        self.gamma2 = gamma2

        # resestimate A,B,self.pi
        # reestimate self.pi
        self.pi = gamma2[0,:]

        # re-estimate A
        for i in range(N):
            for j in range(N):
                numer,denom = 0.0,0.0

                numer = np.sum([gamma[t,i,j] for t in range(T-1)])
                denom = np.sum([gamma2[t,i] for t in range(T-1)])

                self.A[i,j] = numer/denom

        # re-estimate B
        for i in range(N):
            for j in range(M):
                numer,denom = 0.0,0.0

                for t in range(T):
                    if self.Obs[t] == j:
                        numer += gamma2[t,i]
                    denom += gamma2[t,i]
                self.B[i,j] = numer/denom

        # compute log[P(Obs|lambda)]
        logProb = -np.sum([np.log(val) for val in c])
        self.newLogProb = logProb

    def viterbi(self):
        N,M = self.B.shape
        T = len(self.Obs)
        V = np.zeros((T,N)) # V is really alpha from above

        #Initialization is weird
        path = {}
        for i in range(N):
            path[i] = [str(i)]
        V[0,:] = [self.pi[i]*self.B[i,self.Obs[0]] for i in range(N)]

        for t in range(1,T):
            newpath = {}
            for i in range(N):
                (prob, state) = max([(V[t-1,j]*self.A[j,i]*self.B[i,self.Obs[t]], j) for j in range(N)])
                V[t,i] = prob
                newpath[i] = path[state] + [str(i)]
            path = newpath # Don't need to remember the old paths
        (prob, state) = max([(V[T-1][j], j) for j in range(N)])

        return (prob, path[state])

    def get_brown_corpus_observations(self,nmbr_obs = 5000):
        brown = []
        with open('brown.txt','r') as f:
            for line in f:
                lower_line = line.lower()
                final_line = re.sub(r'([^\s\w]|_)+', '', lower_line).strip()
                if final_line == '':
                    continue
                final_line = ' '.join(final_line.split())
                final_line = ''.join(i for i in final_line if not i.isdigit())
                brown.append(final_line)

        output = []
        while (len(output) < nmbr_obs):
            for i,line in enumerate(brown):
                for character in line:
                    if character == ' ':
                        output.append(26)
                    else:
                        output.append(ord(character) - 97)
                output.append(26)
        return output

HMM_obj = HMM()

startTime = int(round(time.time() * 1000))
# Toy Problem
A = np.array([0.7,0.3,0.4,0.6]).reshape(2,2)
B = np.array([0.1,0.4,0.5,0.7,0.2,0.1]).reshape(2,3)
pi = np.array([0.6,0.4])
Obs = np.array([1,1,2])
results = HMM_obj.HMM(A,B,pi,Obs)


print "A:\n",results.A
print "\nB:\n",results.B
print "\npi:\n",results.pi
print "\ngamma2:",results.gamma2
print "\nalpha",results.alpha

# test against two-state english language example
N,M = 2,27
# these should not be uniform.
# make them row stochastic, etc.
A = np.array([0.47468,0.52532,0.51656,0.48344]).reshape(2,2)
B = np.ones((N,M))*HMM_obj.get_stochastic_row_for_initialization(M)
pi = np.array([0.51316,0.48684])
Obs =  np.array(HMM_obj.get_brown_corpus_observations())#corpus construction
results = HMM_obj.HMM(A,B,pi,Obs)

print results.__dict__
endTime = int(round(time.time() * 1000))
print endTime-startTime,'ms'
