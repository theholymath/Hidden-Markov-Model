import numpy as np

def _get_stochastic_row_for_initialization(N):
    eps = (1./N)*(1/100.0)
    if N%2:
        row = [1./N + (-1.0)**i*eps for i in range(N-1)] + [1./N]
    else:
        row = [1./N + (-1.0)**i*eps for i in range(N)]

    return row

    # based on  "A Revealing Introduction to Hidden Markov Models"

N,M = 5,4
# these should not be uniform.
# make them row stochastic, etc.
A = np.ones((N,N))*_get_stochastic_row_for_initialization(N)
B = np.ones((N,M))*_get_stochastic_row_for_initialization(M)
pi = np.ones((N))*_get_stochastic_row_for_initialization(N)

maxIters = 100
iters = 0
oldLogProb = -np.inf

@profile
def _compute_params(A,B,pi,Obs):
    N,M = B.shape
    T = len(Obs)

    c = [0.0]*T
    alpha  = np.ones((T,N))
    beta   = np.ones((T,N))
    gamma  = np.ones((T,N,N))
    gamma2 = np.ones((T,N))

    # compute a_0(i)
    alpha[0,:] = [pi[i]*B[i,int(Obs[0])] for i in range(N)]
    c[0] = np.sum(alpha[0,:])

    # scale a_0[i]
    c[0] = 1.0/c[0]
    alpha[0,:] = [c[0]*alpha[0,i] for i in range(N)]

    # compute a_t(i)
    for t in range(1,T):
        for i in range(N):
            alpha[t,i] = 0.0

            for j in range(N):
                alpha[t,i] += alpha[t-1,j]*A[j,i]


            alpha[t,i] = alpha[t,i]*B[i,Obs[t]]
            c[t] += alpha[t,i]

            # scale alpha[t,i]
            c[t] = 1.0/c[t]

            for i in range(N):
                alpha[t,i] = c[t]*alpha[t,i]


    #### BETA
    beta[T-1,:] = [c[T-1]]*N

    # beta pass
    for t in range(T-2,-1,-1):
        for i in range(N):
            beta[t,i] = 0.0

            for j in range(N):
                beta[t,i] += A[i,j]*B[j,Obs[t+1]]*beta[t+1,j]

            #improve
            beta[t,i] = np.sum([A[i,j]*B[j,Obs[t+1]]*beta[t+1,j] for j in range(N)])

            #scale by same factor
            beta[t,i] = c[t]*beta[t,i]

    # gamma
    for t in range(T-1):
        denom = 0.0
        for i in range(N):
            for j in range(N):
                denom += alpha[t,i]*A[i,j]*B[j,Obs[t+1]]*beta[t+1,j]

            # improve
            denom = np.sum([alpha[t,i]*A[i,j]*B[j,Obs[t+1]]*beta[t+1,j] for j in range(N)])

        for i in range(N):
            gamma2[t,i] = 0.0
            for j in range(N):
                gamma[t,i,j] = (alpha[t,i]*A[i,j]*B[j,Obs[t+1]]*beta[t+1,j])/denom
                gamma2[t,i] += gamma[t,i,j]

            #improve
            gamma2[t,i] = np.sum([gamma[t,i,j] for j in range(N)])


    # special case
    denom = np.sum(alpha[T-1,:])
    gamma2[T-1,:] = [alpha[T-1,i]/denom for i in range(N)]

    # resestimate A,B,pi
    # reestimate pi
    pi = gamma2[0,:]

    # re-estimate A
    for i in range(N):
        for j in range(N):
            numer,denom = 0.0,0.0

            for t in range(T-1):
                numer += gamma[t,i,j]
                denom += gamma2[t,i]
            A[i,j] = numer/denom

    # re-estimate B
    for i in range(N):
        for j in range(M):
            numer,denom = 0.0,0.0

            for t in range(T):
                if Obs[t] == j:
                    numer += gamma2[t,i]
                denom += gamma2[t,i]
            B[i,j] = numer/denom

    # compute log[P(Obs|lambda)]
    logProb = -np.sum([np.log(val) for val in c])
    return logProb,pi,A,B

@profile
def HMM(A,B,pi,Obs,maxIters=100,oldLogProb=-np.inf):
    iters = 0
    logProb,pi,A,B = _compute_params(A,B,pi,Obs)

    while (iters < maxIters and logProb > oldLogProb):
        oldLogProb = logProb
        logProb,pi,A,B = _compute_params(A,B,pi,Obs)
        iters+=1
#     if (iters < maxIters and logProb > oldLogProb):
#         oldLogProb = logProb
#         logProb,pi,A,B = _compute_params(A,B,pi,Obs)
#     else:
    return pi,A,B
N,M = 5,4
# these should not be uniform.
# make them row stochastic, etc.
A = np.ones((N,N))*_get_stochastic_row_for_initialization(N)
B = np.ones((N,M))*_get_stochastic_row_for_initialization(M)
pi = np.ones((N))*_get_stochastic_row_for_initialization(N)

maxIters = 100
iters = 0
oldLogProb = -np.inf
Obs = [1,1,0,3]
#_compute_params(A,B,pi,Obs)
HMM(A,B,pi,Obs)
