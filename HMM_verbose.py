import time
import numpy as np
import re

A = np.array([0.7,0.3,0.4,0.6]).reshape(2,2)
B = np.array([0.1,0.4,0.5,0.7,0.2,0.1]).reshape(2,3)
pi = np.array([0.6,0.4])
Obs = np.array([1,1,2])

startTime = int(round(time.time() * 1000))
N,M = B.shape
T = len(Obs)

c = [0.0]*T
alpha  = np.zeros((T,N))
beta   = np.zeros((T,N))
digamma  = np.zeros((T-1,N,N))
gamma = np.zeros((T,N))

'''
    A more verbose (readable) version of the HMM in HMM.py
    Not a class structure for easier debugging and testing.
    This will be slower
'''
def compute_normalization(alpha):
    return [1./sum(alpha[t,:]) for t in range(T)]

def scale_alpha_beta(alpha,beta,c):
    for t in range(T):
        alpha[t,:] = alpha[t,:]*c[t]
        beta[t,:] = beta[t,:]*c[t]
    return alpha,beta

def get_alpha(A,B,pi,Obs):
    alpha[0,:] = [pi[i]*B[i,int(Obs[0])] for i in range(N)]

    for t in range(1,T):
        for i in range(N):
            alpha[t,i] = np.sum([alpha[t-1,j]*A[j,i] for j in range(N)])
            alpha[t,i] = alpha[t,i]*B[i,Obs[t]]

    return alpha

def get_beta(A,B,pi,Obs):
    beta[T-1,:] = [1.0]*N

    for t in range(T-2,-1,-1):
        for i in range(N):
            beta[t,i] = np.sum([A[i,j]*B[j,Obs[t+1]]*beta[t+1,j] for j in range(N)])
    return beta

def get_digamma(A,B,pi,Obs,alpha,beta):# gamma
    for t in range(T-1):
        denom = 0.0
        for i in range(N):
            for j in range(N):
                denom += alpha[t,i]*A[i,j]*B[j,Obs[t+1]]*beta[t+1,j]

        for i in range(N):
            for j in range(N):
                digamma[t,i,j] = (alpha[t,i]*A[i,j]*B[j,Obs[t+1]]*beta[t+1,j])/denom
    return digamma

def get_gamma(digamma,alpha):
    for t in range(T-1):
        gamma[t,:] = [np.sum([digamma[t,i,:]]) for i in range(N)]

    ## special case T-1
    denom = np.sum(alpha[T-1,:])
    gamma[T-1,:] = [alpha[T-1,i]/denom for i in range(N)]

    return gamma

def reestimate_model(A,B,pi,Obs,digamma,gamma):# resestimate A,B,pi

    # reestimate pi
    pi = gamma[0,:]

    # re-estimate A
    for i in range(N):
        for j in range(N):
            numer,denom = 0.0,0.0

            for t in range(T-1):
                numer += digamma[t,i,j]
                denom += gamma[t,i]
            A[i,j] = numer/denom

    # re-estimate B
    for i in range(N):
        for j in range(M):
            numer,denom = 0.0,0.0

            for t in range(T):
                if Obs[t] == j:
                    numer += gamma[t,i]
                denom += gamma[t,i]
            B[i,j] = numer/denom
    return A,B,pi
def get_log_prob(c):
    return -np.sum([np.log(val) for val in c])

def compute_params(A,B,pi,Obs):
    alpha      = get_alpha(A,B,pi,Obs)
    beta       = get_beta(A,B,pi,Obs)
    c          = compute_normalization(alpha)
    alpha,beta = scale_alpha_beta(alpha,beta,c)
    digamma    = get_digamma(A,B,pi,Obs,alpha,beta)
    gamma      = get_gamma(digamma,alpha)
    A,B,pi     = reestimate_model(A,B,pi,Obs,digamma,gamma)
    newLogProb = get_log_prob(c)

    return A,B,pi,newLogProb

def compute_model(A,B,pi,Obs,maxIters=100):
  oldLogProb = -np.inf # may want to keep track
  iters = 0
  A,B,pi,newLogProb = compute_params(A,B,pi,Obs)

  while (iters < maxIters and newLogProb > oldLogProb):
      oldLogProb = newLogProb
      A,B,pi,newLogProb = compute_params(A,B,pi,Obs)
      iters+=1
  return A,B,pi,newLogProb

A,B,pi,newLogProb = compute_model(A,B,pi,Obs)
endTime = int(round(time.time() * 1000))

print "Total computation time:",endTime - startTime,'ms'
print "A:\n",A
print "\nB:\n",B
print "\npi:\n",pi
