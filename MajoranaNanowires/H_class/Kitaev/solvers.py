
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                  "H_class/Kitaev/solvers" submodule
                      
This sub-package builds Kitaev Hamiltonians for nanowires. Please, visit
http://www.samdaz/MajoranaNanowires.com for more details.


###############################################################################
           
'''


#%%############################################################################
########################    Required Packages      ############################   
###############################################################################
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import multiprocessing

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.constants as cons

from MajoranaNanowires.Functions import order_eig, length, diagonal, H_rectangular2hexagonal, U_hexagonal2rectangular, concatenate




#%%   


###############################################################################
########################    Kitaev Nanowires      #############################   
###############################################################################


#######################    Calling functions     ##############################



            
#%%   
def Kitaev_1D_solver(H,n,mu=0,n_eig='none'):
    
    """
    1D Kitaev Hamiltonian solver. It solves the Hamiltonian of a 1D Kitaev 
    chain.
    
    Parameters
    ----------
        H: arr
            Kitaev Hamiltonian built with Kitaev_builder().
            
        n: int
            Number of times you want to diagonalize the Hamiltonian. In each
            step, it is expected that mu, B or aR is different.
            
        mu: float or arr
            -If mu is a float, the Hamiltonian is diagonalized once adding this
            value to the Hamiltonian
            -If mu is a 1-D array of length=N, the Hamiltonian is diagonalized 
            once adding each element of mu to each site of the built 
            Hamiltonian.
            -If mu is a 1-D array of length=n, the Hamiltonian is diagonalized 
            n times, adding in each step i, the same chemical potential mu[i]
            in every site.
            -If mu is a 2-D array (n x N), the Hamiltonian is diagonalized n 
            times, adding to the Hamiltonian in each step i the chemical
            potential mu[i,:].
            
        n_eig: int
            Number of desire eigenvalues (if H is sparse). 'none' means all.
           
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    #Obtain the dimensions:
    N=int(len(H)/2)
    
    m=n_eig
    
    #Ensure mu and Δ are arrays:
    if (n==1):
        if np.isscalar(mu):
            mu = mu * np.ones(N)
    else:
        if np.isscalar(mu[0]):
            mu_temp=np.zeros((n,N))
            for i in range(n):
                mu_temp[i,:] = mu[i] * np.ones(N)
            mu=mu_temp
            
    #Storing matrices
    E = np.empty([int(m), int(n)])
    U = np.empty([2*N,int(m),n])
    
    #Non-sparse matrices:
    if (n_eig==2*N):
        
        #For just one point:
        if (n==1):
            for i in range(N):
                H[i,i] = H[i,i]-mu[i]
                H[i+N,i+N] = H[i,i]+mu[i]

            #Eigenspectra
            E[:,0], U[:,:,0] = scipy.linalg.eigh(H, lower=False)
            
            return (E[:,0]), (U[:,:,0])
            
            
        #For several points:
        else:   
            for k in range(n):
                for i in range(N):
                    H[i,i] =  H[i,i] -mu[k,i]
                    H[i+N,i+N] = H[i,i] + mu[k,i]
                    
                #Eigenspectra
                E[:, k], U[:,:,k] = scipy.linalg.eigh(H, lower=False)
            
            return (E), (U)
    
    #Sparse matrices:
    else:
        assert n_eig==((not(2*N)) or (not(0))), 'n_eig!=2*N is not still available.'





