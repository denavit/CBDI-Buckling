import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from math import inf,cos,pi,sin,sqrt
from CBDI import hMatrix,gMatrix

plt.rc('font',family='serif')
plt.rc('mathtext',fontset='dejavuserif')
plt.rc('axes',labelsize=8)
plt.rc('axes',titlesize=8)
plt.rc('legend',fontsize=8)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

## Weakened Column
EIo = 1.0
L = 1.0

#alphaArray = [1,0.5,0.1,0.05,0.01]
kArray = [inf,5.0,1.0,0.5,0.01]


## Wang et al. 2004
def g(α,a,k):
    # α is the buckling load parameter
    # a is the relative location of the junction
    # k is the relative stiffness of the junction
    if α < 0:
        return float('nan')
    sα = sqrt(α)
    g = cos(sα) - cos((1-2*a)*sα) + 2/sα*k*sin(sα) # Eq 10 from Wang et al. 2004
    return g

num_k = len(kArray)
num_a = 100
aArray = np.linspace(0.0,0.5,num_a)
psiWang = np.zeros((num_k,num_a))
for ik,k in enumerate(kArray):
    for ia,a in enumerate(aArray):
        
        # Check for quick return
        if k == inf:
            psiWang[ik,ia] = pi**2
            continue
        
        root = fsolve(g,0.01,args=(a,k))
        psiWang[ik,ia] = root[0]


## CBDI Approach
NpArray = [6,10,14]
for Np in NpArray:
    
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_axes([0.13,0.15,0.84,0.82])

    for ik,k in enumerate(kArray):
        if k == inf:
            alpha = 1.0
        else:
            alpha = 1/(Np*(1/Np+1/k))
        
        psiComputed = []
        for iNp in range(Np):

            x = []
            for i in range(Np):
                x.append((0.5+i)/Np)

            h = hMatrix(x,Np)
            g = gMatrix(x,Np)
            lstar = np.dot(h,np.linalg.inv(g))
              
            F = np.zeros((Np,Np))
            for i in range(Np):
                F[i,i] = 1/EIo
            F[iNp,iNp] = 1/(alpha*EIo)

            [v,d] = np.linalg.eig(np.dot(-L**2*lstar,F))
            psiComputed.append(min(1/v))
            
        plt.plot(aArray,psiWang[ik,:],'r-')
        plt.plot(np.append(0,x),np.append(pi**2,psiComputed),'k--x',label=f'k = {k}')
        
    #plt.legend()
    plt.xlabel('Normalized Location of Weakened Section, $a/L$')
    plt.ylabel('Stability variable, $\psi^2$')
    plt.xlim([0,0.5])
    plt.ylim([0,10.5])
    #plt.title(f'$N_p={Np}$ Gauss Integration Points')
    
    figure_name = {6:'a',10:'b',14:'c'}
    plt.savefig(f'Figure_X{figure_name[Np]}_Weakened.png',dpi=300)
    plt.show()