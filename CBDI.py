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