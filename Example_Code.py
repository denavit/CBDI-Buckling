
if __name__ == "__main__":

    import numpy as np
    Np = 5
    x  = [0.1,0.3,0.5,0.7,0.9]
    EI = 1000.0
    L  = 10.0

    # Form h matrix
    h = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            h[i,j] = (x[i]**(j+2)-x[i])/((j+1)*(j+2))

    # Form g (Vandermonde) matrix
    g = np.zeros((Np,Np))
    for j in range(Np):
        for i in range(Np):
            g[i,j] = x[i]**j

    # Form flexibility matrix
    F = np.zeros((Np,Np))
    for i in range(Np):
        F[i,i] = 1/EI

    lstar = np.dot(h,np.linalg.inv(g))
    [v,d] = np.linalg.eig(np.dot(-L**2*lstar,F))

    Pcr = 1/v[0]


    from math import pi
    print(f'Pcr (CBDI)  = {Pcr}')
    print(f'Pcr (Exact) = {pi**2*EI/L**2}')