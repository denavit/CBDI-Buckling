import numpy as np

### Function to define h matrix
def hMatrix (x,Np):
    h = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            h[i,j] = (x[i]**(j+2)-x[i])/((j+1)*(j+2))
            
    return h
    
### Function to define g (Vandermonde) matrix
def gMatrix (x,Np):
    g = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            g[i,j] = x[i]**j
            
    return g
    
### Function for complete buckling analysis, returns only the first buckling mode
def PcrCBDI(x,EI,L):
    Np = len(x)
    
    # h matrix
    h = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            h[i,j] = (x[i]**(j+2)-x[i])/((j+1)*(j+2))
    
    # g matrix
    g = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            g[i,j] = x[i]**j
            
    lstar = np.dot(h,np.linalg.inv(g))
    
    F = np.zeros((Np,Np))
    for i in range(Np):
        F[i,i] = 1/EI[i]
    
    [v,d] = np.linalg.eig(np.dot(-L**2*lstar,F))    
    
    return min(1/v)