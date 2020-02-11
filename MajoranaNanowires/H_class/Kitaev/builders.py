
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                 "H_class/Kitaev/builders" submodule
                      
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

########################     XD functions     ################################
#%%   
## 1D Kitaev Chain:
def Kitaev_1D_builder(N,mu,t,Δ, sparse='no'):
    
    """
    1D Kitaev Hamiltonian builder. It obtaines the Hamiltoninan for a 1D Kitaev
    chain.
    
    Parameters
    ----------
        N: int
            Number of sites.
        
        mu: float or arr
            Chemical potential. If it is an array, each element is the chemical
            potential on each site of the lattice.
            
        t: float
            Hopping elements between sites. t[N] is not used.
            
        Δ: float or arr
            Superconductor hopping element between sites. If it is an array, 
            each element is the hopping on each site of the lattice. Δ(N) is
            not used.
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
    Returns
    -------
        H: arr
            Hamiltonian matrix.
      
    """
    
    #Ensure mu, Δ and t are onsite arrays:
    if np.isscalar(mu):
        mu = mu * np.ones(N)
    if np.isscalar(Δ):
        Δ = Δ * np.ones(N-1)
    if np.isscalar(t):
        t = t * np.ones(N-1)

    #Built the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(2 * N), int(2 * N)))
    elif sparse=='yes':
        H=scipy.sparse.dok_matrix((int(2*N),int(2*N)))
        
    for i in range(N):
        
        H[i,i] = -mu[i]
        
        if i > 0:
            H[i,i-1] = -t[i-1]
            H[i-1,i] = -t[i-1]
            H[i+N,i-1+N] = t[i-1]
            H[i-1+N,i+N] = t[i-1]
            
            H[i,i-1+N] = -Δ[i-1]
            H[i-1+N,i] = -Δ[i-1]
            H[i-1,i+N] = Δ[i-1]
            H[i+N,i-1] = Δ[i-1]
            
    return (H)
            
