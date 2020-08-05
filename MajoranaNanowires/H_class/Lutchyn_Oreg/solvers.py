
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2020)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
              "H_class/Lutchyn_Oreg/builders" submodule
                      
This sub-package solves Lutchyn-Oreg Hamiltonians.

###############################################################################
           
'''


#%%############################################################################
########################    Required Packages      ############################   
###############################################################################
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.constants as cons

from MajoranaNanowires.Functions import order_eig, diagonal
from MajoranaNanowires.Functions import H_rec2shape, U_shape2rec


#%%
def LO_1D_solver(H,N,dis,
                      mu=0,B=0,aR=0,d=0,
                      space='position',k_vec=0,
                      sparse='no',n_eig=None,near=None):
    
    """
    1D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 1D Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 1D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: int
            Number of sites.
            
        dis: int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 1D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 2D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
            
        
        d: float or arr
            On-site superconductivity. If it is float, the SC pairing amplitude
            is the same in every site, while if it is a 1D array, it is the
            on-site superconductivity.
            
                                    
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    
    #Make sure that some parameters are arrays:            
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones(N)
        
    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)

    if np.ndim(aR)==0:
        aRy=np.zeros(N)
        aRz=aR*np.ones(N)/(2*dis)
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRy=aR[1]*np.ones(N)/(2*dis)
            aRz=aR[2]*np.ones(N)/(2*dis)
        else:
            aRy=np.zeros(N)
            aRz=aR/(2*dis)
    else:
        aRy=aR[1]/(2*dis)
        aRz=aR[2]/(2*dis)
        
    if np.isscalar(d):
        d = d * np.ones(N)

    if space=='position' or space=='position2momentum':
        n_k=len(k_vec)

    #Store matrices:
    if n_eig==None:
        n_eig=4*N
        
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([4*N,int(n_eig)],dtype=complex)
    elif space=='momentum'  or space=='position2momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([4*N,int(n_eig),n_k],dtype=complex)
    
    if sparse=='no':
        H_add=np.zeros((4*N,4*N),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H)
        H_add=scipy.sparse.dok_matrix((4*N,4*N),dtype=complex)
            
    #Obtain the add-values Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        for i in range(2):
            H_add[diagonal(2*N*(i+1),init=2*N*i)] = (-1)**i*(np.repeat(e,2))

    if not(np.isscalar(B) and B==0):
        Bz = np.repeat(Bz,2)
        Bz[1::2] = -Bz[::2]
        for i in range(2):
            H_add[diagonal(2*N*(i+1),init=2*N*i,k=1,step=2)], H_add[diagonal(2*N*(i+1),init=2*N*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
            H_add[diagonal(2*N*(i+1),init=2*N*i)] += (-1)**i*Bz
    
    if not((aRy==0).all() and (aRz==0).all()):
        aRy = np.repeat(aRy,2)
        aRy[1::2] = -aRy[::2]
        for i in range(2):
            H_add[diagonal(2*N*(i+1),init=2*N*i,k=-2)]= +1j*aRy[2::]
            H_add[diagonal(2*N*(i+1),init=2*N*i,k=2)] = -1j*aRy[2::]
            H_add[diagonal(2*N*(i+1),k=1,step=2,init=1+2*N*i)] += -1*(-1)**i*aRz[1::]
            H_add[diagonal(2*N*(i+1),k=-1,step=2,init=1+2*N*i)] += -1*(-1)**i*aRz[1::]
            H_add[diagonal(2*N*(i+1),init=2*N*i,k=3,step=2)] += (-1)**i*aRz[1::]
            H_add[diagonal(2*N*(i+1),init=2*N*i,k=-3,step=2)] += (-1)**i*aRz[1::]

    if not(np.isscalar(d)):
        d=d.flatten()
        H_add[diagonal(4*N,k=2*N+1,step=2)], H_add[diagonal(4*N,k=-2*N-1,step=2)] = -np.conj(d), -d
        H_add[diagonal(4*N,k=2*N-1,step=2,init=1)], H_add[diagonal(4*N,k=-2*N+1,step=2,init=1)] = np.conj(d), d
    
    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        if space=='position':
            E[0:2 * N], U[0:4 * N, 0:2 * N] = scipy.linalg.eigh(H+H_add, lower=False,eigvals=(2*N,4*N-1))
            E[2*N:4*N]=-E[0:2*N]
            U[0:2 * N, 2 * N:4 * N] = U[2 * N:4 * N, 0:2 * N]
            U[2 * N:4 * N, 2 * N:4 * N] = U[0:2 * N, 0:2 * N]
            E,U=order_eig(E,U,sparse='no')
            
        elif space=='momentum':
            for i in range(n_k):
                H_add[2 * (N - 1):2 * (N - 1) + 2, 0: 2] = np.array([[-1j*aRy[2], aRz[1]], [-aRz[1], +1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
                H_add[2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N, 2*N: 2+2*N] = -np.array([[+1j*aRy[2], aRz[1]], [-aRz[1], -1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)
                H_add[0: 2, 2 * (N - 1):2 * (N - 1) + 2] = np.array([[+1j*aRy[2], -aRz[1]], [aRz[1], -1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)
                H_add[2*N: 2+2*N, 2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N] = -np.array([[-1j*aRy[2], -aRz[1]], [aRz[1], +1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
        
                E[0:2 * N,i], U[0:4 * N, 0:2 * N,i] = scipy.linalg.eigh(H[:,:,i]+H_add, lower=False,eigvals=(2*N,4*N-1))
                E[2*N:4*N,i]=-E[0:2*N,i]
                U[0:2 * N, 2 * N:4 * N,i] = U[2 * N:4 * N, 0:2 * N,i]
                U[2 * N:4 * N, 2 * N:4 * N,i] = U[0:2 * N, 0:2 * N,i]
                E[:,i],U[:,:,i]=order_eig(E[:,i],U[:,:,i],sparse='no')
        
    else:
        if space=='position':
            E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig,sigma=0, which='LM',tol=1e-5)
            E,U=order_eig(E,U,sparse='yes')
            
        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((4*N,4*N),dtype=complex)
            for i in range(n_k):
                H_k = (H+H_add).copy()
                H_k[0: 2, 2 * (N - 1):2 * (N - 1) + 2] += H_k[2:4, 0: 2]*np.exp(-1j*k_vec[i]*N)
                H_k[2 * (N - 1):2 * (N - 1) + 2, 0: 2] += H_k[0:2,2:4]*np.exp(1j*k_vec[i]*N)
                H_k[2*N: 2+2*N, 2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N] = -(np.conj(H_k[0: 2, 2 * (N - 1):2 * (N - 1) + 2]))
                H_k[2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N, 2*N: 2+2*N] = -(np.conj(H_k[2 * (N - 1):2 * (N - 1) + 2, 0: 2]))
                
                E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig,sigma=0, which='LM',tol=1e-5)
                E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='yes')
            
    #Return the eigenspectra:
    return (E), (U)




#%%
def LO_1D_solver_NoSC(H,N,dis,
                           mu=0,B=0,aR=0,
                           space='position',k_vec=0,
                           sparse='no',n_eig=None,near=None):

    """
    1D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 1D Lutchy-Oreg chain without superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 1D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: int
            Number of sites.
            
        dis: int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 1D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 2D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
                        
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    #Make sure that the onsite parameters are arrays:    if np.isscalar(mu):
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones(N)
        
    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)

    if np.ndim(aR)==0:
        aRy=np.zeros(N)
        aRz=aR*np.ones(N)/(2*dis)
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRy=aR[1]*np.ones(N)/(2*dis)
            aRz=aR[2]*np.ones(N)/(2*dis)
        else:
            aRy=np.zeros(N)
            aRz=aR/(2*dis)
    else:
        aRy=aR[1]/(2*dis)
        aRz=aR[2]/(2*dis)
        
    if space=='position' or space=='position2momentum':
        n_k=len(k_vec)
        
    #Store matrices:
    if n_eig==None:
        n_eig=2*N
        
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([2*N,int(n_eig)],dtype=complex)
    elif space=='momentum'  or space=='position2momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([2*N,int(n_eig),n_k],dtype=complex)
    
    if sparse=='no':
        H_add=np.zeros((2*N,2*N),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H)
        H_add=scipy.sparse.dok_matrix((2*N,2*N),dtype=complex)
            
    #Obtain the add-values Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        H_add[diagonal(2*N)]=np.repeat(e,2)

    if not(np.isscalar(B) and B==0):
        Bz,Bx,By=np.repeat(Bz,2),np.repeat(Bx,2), 1j*np.repeat(By,2)
        Bx[1::2], By[1::2], Bz[1::2] = 0, 0, -Bz[::2]
        H_add[diagonal(2*N,k=1)], H_add[diagonal(2*N,k=-1)] = Bx[:-1]-By[:-1], Bx[:-1]+By[:-1]
        H_add[diagonal(2*N)]+=Bz

    if not((aRy==0).all() and (aRz==0).all()):
        aRy=-1j*np.repeat(aRy,2)
        aRy[1::2]= -aRy[::2]
        H_add[diagonal(2*N,k=-2)], H_add[diagonal(2*N,k=2)] = -aRy[2::], aRy[2::]
        H_add[diagonal(2*N,k=1,step=2,init=1)] += -aRz[1::]
        H_add[diagonal(2*N,k=-1,step=2,init=1)] += -aRz[1::]
        H_add[diagonal(2*N,k=3,step=2)] += aRz[1::]
        H_add[diagonal(2*N,k=-3,step=2)] += aRz[1::]
    
    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        if space=='position':
            E, U= scipy.linalg.eigh(H+H_add, lower=False)
            E,U=order_eig(E,U,sparse='no')
            
        elif space=='momentum':
            for i in range(n_k):
                H_add[2 * (N - 1):2 * (N - 1) + 2, 0: 2] = np.array([[-1j*aRy[2], aRz[1]], [-aRz[1], +1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
                H_add[0: 2, 2 * (N - 1):2 * (N - 1) + 2] = np.array([[+1j*aRy[2], -aRz[1]], [aRz[1], -1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)     
                E[:,i], U[:,:,i] = scipy.linalg.eigh(H[:,:,i]+H_add, lower=False)
                E[:,i],U[:,:,i]=order_eig(E[:,i],U[:,:,i],sparse='no')
    else:
        if space=='position':
            E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig, which='SA',tol=1e-5)
            E,U=order_eig(E,U,sparse='yes',BdG='no')
            
        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((2*N,2*N),dtype=complex)
            for i in range(n_k):
                H_k = (H+H_add).copy()
                H_k[0: 2, 2 * (N - 1):2 * (N - 1) + 2] += H_k[2:4, 0: 2]*np.exp(-1j*k_vec[i]*N)
                H_k[2 * (N - 1):2 * (N - 1) + 2, 0: 2] += H_k[0:2,2:4]*np.exp(1j*k_vec[i]*N)
                E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='SA',tol=1e-5)
                E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='yes',BdG='no')
                
    return (E), (U)

        

        


        
#%%
def LO_2D_solver(H,N,dis,m_eff=0.023,
                      mu=0,B=0,aR=0,d=0,
                      space='position',k_vec=0,
                      sparse='yes',n_eig=None,near=None,
                      section='rectangular'):

    """
    2D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 2D Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 1D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: arr
            Number of sites in each direction.
            
        dis: arr or int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 2D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 2D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 3D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
            
        d: float or arr
            On-site superconductivity. If it is float, the SC pairing amplitude
            is the same in every site, while if it is a 2D array, it is the
            on-site superconductivity.

        dic: numpy array
            Whether to re-use the dictionary of sites of other process or not.
            
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    #Obtain the dimensions:          
    Ny, Nz = N[0], N[1]
    
    if np.ndim(dis)==0:
        dis_y, dis_z = dis, dis
    else: 
        dis_y, dis_z = dis[0], dis[1]
        
    m = int(4 * Ny * Nz)  
    if (np.isscalar(section) and not(section=='rectangular')) or not(np.isscalar(section)):
        m_hex=H_rec2shape(0,section,N,dis,BdG='yes',output='m')
        
    if (space=='momentum'):
        n_k=len(k_vec)
        
    if sparse=='no':
        n_eig=m
        
    #Make sure that the onsite parameters are arrays:
    if space=='momentum':
        if np.isscalar(m_eff):
            m_eff = m_eff * np.ones((Ny,Nz))
        m_eff=m_eff.flatten()
    
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones((Ny,Nz))

    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    
    if np.ndim(aR)==0:
        aRx=np.zeros(N)
        aRy=np.zeros(N)
        aRz=aR*np.ones(N)
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRx=aR[0]*np.ones(N)
            aRy=aR[1]*np.ones(N)
            aRz=aR[2]*np.ones(N)
        else:
            aRx=np.zeros(N)
            aRy=np.zeros(N)
            aRz=aR*np.ones(N)
    else:
        aRx=aR[0]
        aRy=aR[1]
        aRz=aR[2]
        
    if np.isscalar(d) and not(d==0):
        d = d * np.ones(N)

    #Store matrices:
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([m,int(n_eig)],dtype=complex)
    elif space=='momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([m,int(n_eig),n_k],dtype=complex)
    
    if sparse=='no':
        H_add=np.zeros((m,m),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H,dtype=complex)
        H_add=scipy.sparse.dok_matrix((m,m),dtype=complex)
    
    #Obtain the add-values Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        e=e.flatten()
        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] = (-1)**(i)*(np.repeat(e,2))
        
    if not(np.isscalar(B) and B==0):
        Bx,By,Bz=Bx.flatten(),By.flatten(),Bz.flatten()
        Bz=np.repeat(Bz,2)
        Bz[1::2] = -Bz[::2]
        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1,step=2)], H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] += (-1)**(i)*(Bz)
        
    if not((aRx==0).all() and (aRy==0).all() and (aRz==0).all()):
        aRx_ky, aRz_ky = np.repeat(((aRx[1::,:]+aRx[:-1,:])/(4*dis_y)).flatten(),2), ((aRz[1::,:]+aRz[:-1,:])/(4*dis_y)).flatten()
        aRx_kz, aRy_kz = ((aRx[:,1::]+aRx[:,:-1])/(4*dis_z)).flatten(), ((aRy[:,1::]+aRy[:,:-1])/(4*dis_z)).flatten()
        aRx_ky[1::2] = -aRx_ky[::2] 
        aRx_kz, aRy_kz = np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1)))

        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Nz)] = 1j*aRx_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Nz)] = -1j*aRx_ky
            H_add[diagonal(int(m/2)*(i+1),k=2*Nz-1,step=2,init=1+int(m/2)*i)] += -1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),k=-2*Nz+1,step=2,init=1+int(m/2)*i)] += 1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Nz,step=2)] += -1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Nz,step=2)] += 1j*aRz_ky
            
            H_add[diagonal(int(m/2)*(i+1),k=1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz+1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),k=-1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz-1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=3,step=2)] += -1*(-1)**(i)*aRx_kz+1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-3,step=2)] += -1*(-1)**(i)*aRx_kz-1j*aRy_kz
        
    if not(np.isscalar(d)):
        d=d.flatten()
        H_add[diagonal(m,k=int(m/2)+1,step=2)], H_add[diagonal(m,k=-int(m/2)-1,step=2)] = -np.conj(d), -d
        H_add[diagonal(m,k=int(m/2)-1,step=2,init=1)], H_add[diagonal(m,k=-int(m/2)+1,step=2,init=1)] = np.conj(d), d

        
    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        #####revisar
        if space=='position':
            E[0:int(m/2)], U[0:m, 0:int(m/2)] = scipy.linalg.eigh(H+H_add, lower=False,eigvals=(2*N,4*N-1))
            E[int(m/2):m]=-E[0:int(m/2)]
            U[0:int(m/2), int(m/2):m] = U[int(m/2):m, 0:int(m/2)]
            U[int(m/2):m, int(m/2):m] = U[0:int(m/2), 0:int(m/2)]
            E,U=order_eig(E,U,sparse='no')
            
        elif space=='momentum':
            aRy, aRz = aRy.flatten(), aRz.flatten()
            aRy=np.repeat(aRy,2)
            aRy[1::2] = -aRy[::2]
            for i in range(n_k):
                for j in range(2):
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] += (-1)**(j)*np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] += -1*aRy*k_vec[i]
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=1,step=2)] += -1j*(-1)**(j)*aRz*k_vec[i]
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-1,step=2)] += 1j*(-1)**(j)*aRz*k_vec[i]
        
                E[0:int(m/2),i], U[0:m, 0:int(m/2),i] = scipy.linalg.eigh(H+H_add, lower=False,eigvals=(2*N,4*N-1))
                E[int(m/2):m,i]=-E[0:int(m/2),i]
                U[0:int(m/2), int(m/2):m,i] = U[int(m/2):m, 0:int(m/2),i]
                U[int(m/2):m, int(m/2):m,i] = U[0:int(m/2), 0:int(m/2),i]
                E[:,i],U[:,:,i]=order_eig(E[:,i],U[:,:,i],sparse='no')
                
                for j in range(2):
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] -= (-1)**(j)*np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] -= -1*aRy*k_vec[i]
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=1,step=2)] -= -1j*(-1)**(j)*aRz*k_vec[i]
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-1,step=2)] -= 1j*(-1)**(j)*aRz*k_vec[i]
        #####
    else:
        if space=='position':
            if np.isscalar(section) and section=='rectangular':
                E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig,sigma=0, which='LM',tol=1e-4)
                E, U=order_eig(E, U,sparse='yes')
            
            else:
                H=H_rec2shape(H+H_add,section,N,dis,BdG='yes',output='H',m=m_hex)
                E, U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H),k = n_eig, which='LM',sigma=0,tol=1e-9)
                E,U_hex=order_eig(E,U_hex,sparse='yes',BdG='yes')
                U=U_shape2rec(U_hex,section,N,dis,BdG='yes')

        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((m,m),dtype=complex)
            aRy, aRz = aRy.flatten(), aRz.flatten()
            aRy=np.repeat(aRy,2)
            aRy[1::2] = -aRy[::2]
            for i in range(n_k):
                H_k = (H+H_add).copy()
                for j in range(2):
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] += (-1)**(j)*np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j)] += -1*aRy*k_vec[i]
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=1,step=2)] += -1j*(-1)**(j)*aRz*k_vec[i]
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-1,step=2)] += 1j*(-1)**(j)*aRz*k_vec[i]
                
                if np.isscalar(section) and section=='rectangular':
                    E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig,sigma=0, which='LM',tol=1e-5)
                    E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='yes')
                    
                else:
                    H_k=H_rec2shape(H_k,section,N,dis,BdG='yes',output='H',m=m_hex)
                    E[:,i], U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='LM',sigma=0,tol=1e-9)
                    E[:,i],U_hex=order_eig(E[:,i],U_hex,sparse='yes',BdG='yes')
                    U[:,:,i]=U_shape2rec(U_hex,section,N,dis,BdG='yes')


    return (E), (U)




        
#%%
def LO_2D_solver_NoSC(H,N,dis,m_eff=0.023,
                           mu=0,B=0,aR=0,
                           space='position',k_vec=0,
                           sparse='yes',n_eig=None,near=None,
                           section='rectangular'):

    
    """
    2D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 2D Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 2D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: arr
            Number of sites in each direction.
            
        dis: int or int
            Distance (in nm) between sites. 
            
        m_eff: int 
            Effective mass.
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 2D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 3D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
            
        dic: numpy array
            Whether to re-use the dictionary of sites of other process or not.
            
        SC: {}
            If the dictionary is not empty, a Superconductor Hamiltonian is
            added to H before diagonalizing it. The elements of the dictionary...
            
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    #Obtain the dimensions:          
    Ny, Nz = N[0], N[1]
    
    if np.ndim(dis)==0:
        dis_y, dis_z = dis, dis
    else: 
        dis_y, dis_z = dis[0], dis[1]
        
    m = int(2 * Ny * Nz)
    if (np.isscalar(section) and not(section=='rectangular')) or not(np.isscalar(section)):
        m_hex=H_rec2shape(0,section,N,dis,BdG='no',output='m')
        
    if (space=='momentum'):
        n_k=len(k_vec)
        
    if sparse=='no':
        n_eig=m
        
    #Make sure that the onsite parameters are arrays:
    if space=='momentum':
        if np.isscalar(m_eff):
            m_eff = m_eff * np.ones((Ny,Nz))
        m_eff=m_eff.flatten()
    
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones((Ny,Nz))

    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    
    if np.ndim(aR)==0:
        aRx=np.zeros(N)
        aRy=np.zeros(N)
        aRz=aR*np.ones(N)
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRx=aR[0]*np.ones(N)
            aRy=aR[1]*np.ones(N)
            aRz=aR[2]*np.ones(N)
        else:
            aRx=np.zeros(N)
            aRy=np.zeros(N)
            aRz=aR*np.ones(N)
    else:
        aRx=aR[0]
        aRy=aR[1]
        aRz=aR[2]

    #Store matrices:        
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([m,int(n_eig)],dtype=complex)

    elif space=='momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([m,int(n_eig),n_k],dtype=complex)
    
    if sparse=='no':
        H_add=np.zeros((m,m),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H,dtype=complex)
        H_add=scipy.sparse.dok_matrix((m,m),dtype=complex)
           
    #Obtain the add-values Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        e=e.flatten()
        H_add[diagonal(m)] = np.repeat(e,2)
        
    if not(np.isscalar(B) and B==0):
        Bx,By,Bz=Bx.flatten(),By.flatten(),Bz.flatten()
        Bz=np.repeat(Bz,2)
        Bz[1::2] = -Bz[::2]
        
        H_add[diagonal(m,k=1,step=2)], H_add[diagonal(m,k=-1,step=2)] = Bx-1j*By, Bx+1j*By
        H_add[diagonal(m)] += Bz
        
    if not((aRx==0).all() and (aRy==0).all() and (aRz==0).all()):
        aRx_ky, aRz_ky = np.repeat(((aRx[1::,:]+aRx[:-1,:])/(4*dis_y)).flatten(),2), ((aRz[1::,:]+aRz[:-1,:])/(4*dis_y)).flatten()
        aRx_kz, aRy_kz = ((aRx[:,1::]+aRx[:,:-1])/(4*dis_z)).flatten(), ((aRy[:,1::]+aRy[:,:-1])/(4*dis_z)).flatten()
        aRx_ky[1::2] = -aRx_ky[::2] 
                
        H_add[diagonal(m,k=2*Nz)] = 1j*aRx_ky
        H_add[diagonal(m,k=-2*Nz)] = -1j*aRx_ky
        H_add[diagonal(m,k=2*Nz-1,step=2,init=1)] += -1j*aRz_ky
        H_add[diagonal(m,k=-2*Nz+1,step=2,init=1)] += 1j*aRz_ky
        H_add[diagonal(m,k=1+2*Nz,step=2)] += -1j*aRz_ky
        H_add[diagonal(m,k=-1-2*Nz,step=2)] += 1j*aRz_ky
        
        aRx_kz, aRy_kz = np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1)))
        H_add[diagonal(m,k=1,step=2,init=1)] += aRx_kz+1j*aRy_kz
        H_add[diagonal(m,k=-1,step=2,init=1)] += aRx_kz-1j*aRy_kz
        H_add[diagonal(m,k=3,step=2)] += -aRx_kz+1j*aRy_kz
        H_add[diagonal(m,k=-3,step=2)] += -aRx_kz-1j*aRy_kz


    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        if space=='position':
            if np.isscalar(section) and section=='rectangular':
                E, U = scipy.linalg.eigh(H+H_add, lower=False)
                E,U=order_eig(E,U,sparse='no',BdG='no')
                
            else:
                H=H_rec2shape(H+H_add,section,N,dis,BdG='no',output='H',m=m_hex)
                E, U_hex = scipy.linalg.eigh(H, lower=False)
                E,U_hex=order_eig(E,U_hex,sparse='no',BdG='no')
                U=U_shape2rec(U_hex,section,N,dis,BdG='no')
        
        elif space=='momentum':
            aRy, aRz = aRy.flatten(), aRz.flatten()
            aRy=np.repeat(aRy,2)
            aRy[1::2] = -aRy[::2]
            for i in range(n_k):
                H_add[diagonal(m)] += np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                H_add[diagonal(m)] += -1*aRy*k_vec[i]
                H_add[diagonal(m,k=1,step=2)] += -1j*aRz*k_vec[i]
                H_add[diagonal(m,k=-1,step=2)] += 1j*aRz*k_vec[i]
                
                if np.isscalar(section) and section=='rectangular':
                    E[:,i],U[:,:,i] = scipy.linalg.eigh(H+H_add, lower=False)
                    E[:,i],U[:,:,i]=order_eig(E[:,i],U[:,:,i],sparse='no',BdG='no')
                    
                else:                    
                    H_k=H_rec2shape(H+H_add,section,N,dis,BdG='no',output='H',m=m_hex)
                    E[:,i], U_hex = scipy.linalg.eigsh(H_k, lower=False)
                    E[:,i],U_hex=order_eig(E[:,i],U_hex,sparse='no',BdG='no')
                    U[:,:,i]=U_shape2rec(U_hex,section,N,dis,BdG='no')
                
                H_add[diagonal(m)] -= np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                H_add[diagonal(m)] -= -1*aRy*k_vec[i]
                H_add[diagonal(m,k=1,step=2)] -= -1j*aRz*k_vec[i]
                H_add[diagonal(m,k=-1,step=2)] -= 1j*aRz*k_vec[i] 
                
    else:
        if space=='position':
            if np.isscalar(section) and section=='rectangular':
                if not(near==None):
                    E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig, which='LA',sigma=near,tol=1e-9)
                else:
                    E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig, which='SA',tol=1e-9)
                E,U=order_eig(E,U,sparse='yes',BdG='no')
                
            else:
                H=H_rec2shape(H+H_add,section,N,dis,BdG='no',output='H',m=m_hex)
                if not(near==None):
                    E, U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H),k = n_eig, which='LA',sigma=near,tol=1e-9)
                else:
                    E, U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H),k = n_eig, which='SA',tol=1e-9)
                E,U_hex=order_eig(E,U_hex,sparse='yes',BdG='no')
                U=U_shape2rec(U_hex,section,N,dis,BdG='no')
                
        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((m,m),dtype=complex)
            aRy, aRz = aRy.flatten(), aRz.flatten()
            aRy=np.repeat(aRy,2)
            aRy[1::2] = -aRy[::2]
            for i in range(n_k):
                print(i)
                H_k = (H+H_add).copy()             
                H_k[diagonal(m)] += np.repeat(cons.hbar**2/(2*m_eff*cons.m_e*(1e-9)**2)/cons.e*1e3*k_vec[i]**2,2)
                H_k[diagonal(m)] += -1*aRy*k_vec[i]
                H_k[diagonal(m,k=1,step=2)] += -1j*aRz*k_vec[i]
                H_k[diagonal(m,k=-1,step=2)] += 1j*aRz*k_vec[i]
                
                if np.isscalar(section) and section=='rectangular':
                    E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='SA',tol=1e-5)
                    E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='yes',BdG='no')
                    
                else:                    
                    H_k=H_rec2shape(H_k,section,N,dis,BdG='no',output='H',m=m_hex)
                    if not(near==None):
                        E[:,i], U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='LA',sigma=near,tol=1e-4)
                    else:
                        E[:,i], U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='SA',tol=1e-4)
                    E[:,i],U_hex=order_eig(E[:,i],U_hex,sparse='yes',BdG='no')
                    U[:,:,i]=U_shape2rec(U_hex,section,N,dis,BdG='no')
                
                
    return (E), (U)

                    
#%%
def LO_3D_solver(H,N,dis,
                      mu=0,B=0,aR=0,d=0,
                      space='position',k_vec=0,
                      sparse='yes',n_eig=None,near=None,
                      section='rectangular'):
    
    """
    3D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 3D Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 3D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: arr
            Number of sites in each direction.
            
        dis: int or int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 3D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 3D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
        
        d: float or arr
            On-site superconductivity. If it is float, the SC pairign amplitude
            is the same in every site, while if it is a 3D array, it is the
            on-site superconductivity.

        dic: numpy array
            Whether to re-use the dictionary of sites of other process or not.
            
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    
    #Obtain dimensions:          
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
        
    m = int(4 * Nx * Ny * Nz)
    if (np.isscalar(section) and not(section=='rectangular')) or not(np.isscalar(section)):
        m_hex=H_rec2shape(0,section,N,dis,BdG='yes',output='m')
        
    if (space=='momentum'):
        n_k=len(k_vec)

    if sparse=='no':
        n_eig=m
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1 and len(B)==3:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    
    if np.ndim(aR)==0:
        aRx=np.zeros((Nx,Ny,Nz))
        aRy=np.zeros((Nx,Ny,Nz))
        aRz=aR*np.ones((Nx,Ny,Nz))
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRx=aR[0]*np.ones((Nx,Ny,Nz))
            aRy=aR[1]*np.ones((Nx,Ny,Nz))
            aRz=aR[2]*np.ones((Nx,Ny,Nz))
        else:
            aRx=np.zeros((Nx,Ny,Nz))
            aRy=np.zeros((Nx,Ny,Nz))
            aRz=aR*np.ones((Nx,Ny,Nz))
    else:
        aRx=aR[0]
        aRy=aR[1]
        aRz=aR[2]

    
    if np.isscalar(d) and not(d==0):
        d = d * np.ones((Nx,Ny,Nz))

    #Store matrices:
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([m,int(n_eig)],dtype=complex)
    elif space=='momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([m,int(n_eig),n_k],dtype=complex)

    
    if sparse=='no':
        H_add=np.zeros((m,m),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H,dtype=complex)
        H_add=scipy.sparse.dok_matrix((m,m),dtype=complex)
            
    #Build the Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        e=e.flatten()
        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] = (-1)**(i)*np.repeat(e,2)
        
    if not(np.isscalar(B) and B==0):
        Bx,By,Bz=Bx.flatten(),By.flatten(),Bz.flatten()
        Bz=np.repeat(Bz,2)
        Bz[1::2] = -Bz[::2]
        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1,step=2)], H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] = (-1)**(i)*Bz
        
    if not((aRy==0).all() and (aRz==0).all()):
        aRy_kx, aRz_kx = np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
        aRx_ky, aRz_ky = np.repeat(((aRx[:,1::,:]+aRx[:,:-1,:])/(4*dis_y)).flatten(),2), ((aRz[:,1::,:]+aRz[:,:-1,:])/(4*dis_y)).flatten()
        aRx_kz, aRy_kz = ((aRx[:,:,1::]+aRx[:,:,:-1])/(4*dis_z)).flatten(), ((aRy[:,:,1::]+aRy[:,:,:-1])/(4*dis_z)).flatten()
        aRy_kx[1::2], aRx_ky[1::2] = -aRy_kx[::2], -aRx_ky[::2] 
        aRx_ky, aRz_ky = np.insert(aRx_ky,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))),np.insert(aRz_ky,np.repeat(np.arange((Nz*Ny-Nz),(Ny*Nz-Nz)*Nx,(Ny*Nz-Nz)),Nz),np.zeros(Nz*(Nx-1)))
        aRx_kz, aRy_kz = np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1)))

        for i in range(2):
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Ny*Nz)] = -1j*aRy_kx
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Ny*Nz)] = +1j*aRy_kx
            H_add[diagonal(int(m/2)*(i+1),k=2*Ny*Nz-1,step=2,init=1+int(m/2)*i)] += -1*(-1)**(i)*aRz_kx
            H_add[diagonal(int(m/2)*(i+1),k=-2*Ny*Nz+1,step=2,init=1+int(m/2)*i)] += -1*(-1)**(i)*aRz_kx
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Ny*Nz,step=2)] += (-1)**(i)*aRz_kx
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Ny*Nz,step=2)] += (-1)**(i)*aRz_kx
            
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Nz)] = +1j*aRx_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Nz)] = -1j*aRx_ky
            H_add[diagonal(int(m/2)*(i+1),k=2*Nz-1,step=2,init=1+int(m/2)*i)] += -1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),k=-2*Nz+1,step=2,init=1+int(m/2)*i)] += 1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Nz,step=2)] += -1j*aRz_ky
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Nz,step=2)] += 1j*aRz_ky
            
            H_add[diagonal(int(m/2)*(i+1),k=1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz+1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),k=-1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz-1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=3,step=2)] += -1*(-1)**(i)*aRx_kz+1j*aRy_kz
            H_add[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-3,step=2)] += -1*(-1)**(i)*aRx_kz-1j*aRy_kz
    
    if not(np.isscalar(d)):
        d=d.flatten()
        H_add[diagonal(m,k=int(m/2)+1,step=2)], H_add[diagonal(m,k=-int(m/2)-1,step=2)] = -np.conj(d), -d
        H_add[diagonal(m,k=int(m/2)-1,step=2,init=1)], H_add[diagonal(m,k=-int(m/2)+1,step=2,init=1)] = np.conj(d), d

        
    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        ######### revisar
        if space=='position':
            E[0:int(m/2)], U[0:m, 0:int(m/2)] = scipy.linalg.eigh(H+H_add, lower=False,eigvals=(2*N,4*N-1))
            E[int(m/2):m]=-E[0:int(m/2)]
            U[0:int(m/2), int(m/2):m] = U[int(m/2):m, 0:int(m/2)]
            U[int(m/2):m, int(m/2):m] = U[0:int(m/2), 0:int(m/2)]
            E,U=order_eig(E,U,sparse='no')
            
        elif space=='momentum':
            for i in range(n_k):
                for j in range(2):
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m-2*Ny*Nz)] = (-1j*aRy_kx)*np.exp(-1j*k_vec[i]*Nx*(-1)**(i))
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m+2*Ny*Nz)] = (+1j*aRy_kx)*np.exp(1j*k_vec[i]*Nx*(-1)**(i))
                    H_add[diagonal(int(m/2)*(j+1),k=m-2*Ny*Nz-1,step=2,init=1+int(m/2)*j)] = (-1)**(i)*(-aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                    H_add[diagonal(int(m/2)*(j+1),k=-m+2*Ny*Nz+1,step=2,init=1+int(m/2)*j)] = (-1)**(i)*(-aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m+1-2*Ny*Nz,step=2)] = (-1)**(i)*(aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                    H_add[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m-1+2*Ny*Nz,step=2)] = (-1)**(i)*(aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
        
                E[0:int(m/2),i], U[0:m, 0:int(m/2),i] = scipy.linalg.eigh(H[:,:,i]+H_add, lower=False,eigvals=(2*N,4*N-1))
                E[int(m/2):m,i]=-E[0:int(m/2),i]
                U[0:int(m/2), int(m/2):m,i] = U[int(m/2):m, 0:int(m/2),i]
                U[int(m/2):m, int(m/2):m,i] = U[0:int(m/2), 0:int(m/2),i]
                E[:,i],U[:,:,i]=order_eig(E[:,i],U[:,:,i],sparse='no')
        #########
        
    else:
        if space=='position':
            if np.isscalar(section) and section=='rectangular':
                E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig,sigma=0, which='LM',tol=1e-4)
                E, U=order_eig(E, U,sparse='yes') 
                
            else:
                H=H_rec2shape(H+H_add,section,N,dis,BdG='yes',output='H',m=m_hex)
                E, U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H),k = n_eig,sigma=0, which='LM',tol=1e-4)
                E, U_hex=order_eig(E, U_hex,sparse='yes')
                U=U_shape2rec(U_hex,section,N,dis,BdG='yes')
                
        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((m,m),dtype=complex)
            for i in range(n_k):
                H_k = (H+H_add).copy()
                for j in range(2):
                    if not((aRy==0).all()):
                        H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m-2*Ny*Nz)] = (-1j*aRy_kx)*np.exp(-1j*k_vec[i]*Nx*(-1)**(i))
                        H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m+2*Ny*Nz)] = (+1j*aRy_kx)*np.exp(1j*k_vec[i]*Nx*(-1)**(i))
                    if not((aRz==0).all()):
                        H_k[diagonal(int(m/2)*(j+1),k=m-2*Ny*Nz-1,step=2,init=1+int(m/2)*j)] = (-1)**(i)*(-aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                        H_k[diagonal(int(m/2)*(j+1),k=-m+2*Ny*Nz+1,step=2,init=1+int(m/2)*j)] = (-1)**(i)*(-aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                        H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m+1-2*Ny*Nz,step=2)] = (-1)**(i)*(aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                        H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m-1+2*Ny*Nz,step=2)] = (-1)**(i)*(aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx*(-1)**(i))
                    
                if np.isscalar(section) and section=='rectangular':
                    E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig,sigma=0, which='LM',tol=1e-5)
                    E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='yes')
                    
                else:
                    H_k=H_rec2shape(H_k,section,N,dis,BdG='yes',output='H',m=m_hex)
                    E[:,i], U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig,sigma=0, which='LM',tol=1e-5)
                    E[:,i], U_hex =order_eig(E[:,i], U_hex,sparse='yes')
                    U[:,:,i]=U_shape2rec(U_hex,section,N,dis,BdG='yes')

    return (E), (U)


#%%
def LO_3D_solver_NoSC(H,N,dis,
                           mu=0,B=0,aR=0,
                           space='position',k_vec=0,
                           sparse='yes',n_eig=None,near=None,
                           section='rectangular'):

    """
    3D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 3D Lutchy-Oreg chain withot superconductivity.
    
    Parameters
    ----------
        H: arr
            Discretized 3D Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: arr
            Number of sites in each direction.
            
        dis: int or int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 3D array, it is the
            on-site chemical potential.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 3D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
            
        dic: numpy array
            Whether to re-use the dictionary of sites of other process or not.
            
        space: {"position","momentum","position2momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. "position2momentum" means that the Hamiltonian is
            built in real space, but you want to diagonalize it in momentum
            space (so in each step is converted to a momentum space).This
            option is recommended for large matrices.
            
        k_vec: arr
            If space=='momentum' or "position2momentum", k_vec is the 
            (discretized) momentum vector, usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        near: float
            If sparse=="yes" and BdG=='no', near provides the value around to
            which the eigenvalues must be found.
            
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
           
        Rashba={"Full-Rashba","kx-terms"}
            Whether include all the terms of the Rashba coupling (Full-Rashba)
            or include only those terms proportional to kx (kx-terms).
            
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
    """
    
    #Obtain dimensions:
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
        
    m = int(2 * Nx * Ny * Nz)
    if (np.isscalar(section) and not(section=='rectangular')) or not(np.isscalar(section)):
        m_hex=H_rec2shape(0,section,N,dis,BdG='no',output='m')

    if (space=='momentum'):
        n_k=len(k_vec)
        
    if sparse=='no':
        n_eig=m
    
    #Make sure that the onsite parameters are arrays:        
    if np.isscalar(mu) and not(mu==0):
        mu = mu * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(B) and not(B==0):
        Bx=B
        By=0
        Bz=0
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    elif np.ndim(B)==1 and len(B)==3:
        Bx=B[0]
        By=B[1]
        Bz=B[2]
        Bx,By,Bz=Bx*np.ones(N),By*np.ones(N),Bz*np.ones(N)
    
    if np.ndim(aR)==0:
        aRx=np.zeros((Nx,Ny,Nz))
        aRy=np.zeros((Nx,Ny,Nz))
        aRz=aR*np.ones((Nx,Ny,Nz))
    elif np.ndim(aR)==1:
        if len(aR)==3:
            aRx=aR[0]*np.ones((Nx,Ny,Nz))
            aRy=aR[1]*np.ones((Nx,Ny,Nz))
            aRz=aR[2]*np.ones((Nx,Ny,Nz))
        else:
            aRx=np.zeros((Nx,Ny,Nz))
            aRy=np.zeros((Nx,Ny,Nz))
            aRz=aR*np.ones((Nx,Ny,Nz))
    else:
        aRx=aR[0]
        aRy=aR[1]
        aRz=aR[2]

    #Store matrices:
    if space=='position':
        E = np.empty([int(n_eig)])
        U = np.empty([m,int(n_eig)],dtype=complex)
    elif space=='momentum':
        E = np.empty([int(n_eig),n_k])
        U = np.empty([m,int(n_eig),n_k],dtype=complex)
    
    if sparse=='no':
        H_add=np.zeros((m,m),dtype=complex)
    else:
        if not(scipy.sparse.issparse(H)):
            H = scipy.sparse.dok_matrix(H,dtype=complex)
        H_add=scipy.sparse.dok_matrix((m,m),dtype=complex)
            
    #Build the Hamiltonian:
    if not(np.isscalar(mu)):
        e=-mu
        e=e.flatten()
        H_add[diagonal(m)] = np.repeat(e,2)
        
    if not(np.isscalar(B) and B==0):
        Bx,By,Bz=Bx.flatten(),By.flatten(),Bz.flatten()
        Bz=np.repeat(Bz,2)
        Bz[1::2] = -Bz[::2]
        H_add[diagonal(m,k=1,step=2)], H_add[diagonal(m,k=-1,step=2)] = Bx-1j*By, Bx+1j*By
        H_add[diagonal(m)] = + Bz
        
    if not((aRx==0).all() and (aRy==0).all() and (aRz==0).all()):
        aRy_kx, aRz_kx = np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
        aRx_ky, aRz_ky = np.repeat(((aRx[:,1::,:]+aRx[:,:-1,:])/(4*dis_y)).flatten(),2), ((aRz[:,1::,:]+aRz[:,:-1,:])/(4*dis_y)).flatten()
        aRx_kz, aRy_kz = ((aRx[:,:,1::]+aRx[:,:,:-1])/(4*dis_z)).flatten(), ((aRy[:,:,1::]+aRy[:,:,:-1])/(4*dis_z)).flatten()
        aRy_kx[1::2], aRx_ky[1::2] = -aRy_kx[::2], -aRx_ky[::2] 
        
        H_add[diagonal(m,k=2*Ny*Nz)] = -1j*aRy_kx
        H_add[diagonal(m,k=-2*Ny*Nz)] = +1j*aRy_kx
        H_add[diagonal(m,k=2*Ny*Nz-1,step=2,init=1)] += -aRz_kx
        H_add[diagonal(m,k=-2*Ny*Nz+1,step=2,init=1)] += -aRz_kx
        H_add[diagonal(m,k=1+2*Ny*Nz,step=2)] += aRz_kx
        H_add[diagonal(m,k=-1-2*Ny*Nz,step=2)] += aRz_kx
        
        aRx_ky, aRz_ky = np.insert(aRx_ky,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))),np.insert(aRz_ky,np.repeat(np.arange((Nz*Ny-Nz),(Ny*Nz-Nz)*Nx,(Ny*Nz-Nz)),Nz),np.zeros(Nz*(Nx-1)))
        H_add[diagonal(m,k=2*Nz)] = +1j*aRx_ky
        H_add[diagonal(m,k=-2*Nz)] = -1j*aRx_ky
        H_add[diagonal(m,k=2*Nz-1,step=2,init=1)] += -1j*aRz_ky
        H_add[diagonal(m,k=-2*Nz+1,step=2,init=1)] += 1j*aRz_ky
        H_add[diagonal(m,k=1+2*Nz,step=2)] += -1j*aRz_ky
        H_add[diagonal(m,k=-1-2*Nz,step=2)] += 1j*aRz_ky
        
        aRx_kz, aRy_kz = np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1)))
        H_add[diagonal(m,k=1,step=2,init=1)] += aRx_kz+1j*aRy_kz
        H_add[diagonal(m,k=-1,step=2,init=1)] += aRx_kz-1j*aRy_kz
        H_add[diagonal(m,k=3,step=2)] += -aRx_kz+1j*aRy_kz
        H_add[diagonal(m,k=-3,step=2)] += -aRx_kz-1j*aRy_kz


        
    #Diagonalize the Hamiltonian:      
    if sparse=='no':
        ###### revisar:
        if space=='position':
            E, U = scipy.linalg.eigh(H+H_add, lower=False)
            E, U=order_eig(E, U,sparse='no') 
            
        elif space=='momentum':
            for i in range(n_k):
                H_add[diagonal(m,k=m-2*Ny*Nz)] = (-1j*aRy_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_add[diagonal(m,k=-m+2*Ny*Nz)] = (+1j*aRy_kx)*np.exp(1j*k_vec[i]*Nx)
                H_add[diagonal(m,k=m-2*Ny*Nz-1,step=2,init=1)] = (-aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_add[diagonal(m,k=-m+2*Ny*Nz+1,step=2,init=1)] = (-aRz_kx)*np.exp(1j*k_vec[i]*Nx)
                H_add[diagonal(m,k=m+1-2*Ny*Nz,step=2)] = (aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_add[diagonal(m,k=-m-1+2*Ny*Nz,step=2)] = (aRz_kx)*np.exp(1j*k_vec[i]*Nx)
                
                E[:,i], U[:,:,i]= scipy.linalg.eigh(H[:,:,i]+H_add, lower=False)
                E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='no') 
        ########
            
    else:
        if space=='position':
            if np.isscalar(section) and section=='rectangular':
                E, U = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H+H_add),k = n_eig, which='SA',tol=1e-5)
                E, U=order_eig(E, U,sparse='yes',BdG='no')
            else:
                H=H_rec2shape(H+H_add,section,N,dis,BdG='no',output='H',m=m_hex)
                E, U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H),k = n_eig, which='SA',tol=1e-5)
                E, U_hex=order_eig(E, U_hex,sparse='yes',BdG='no')
                U=U_shape2rec(U_hex,section,N,dis,BdG='no')


        elif space=='momentum':
            H_k= scipy.sparse.dok_matrix((m,m),dtype=complex)
            for i in range(n_k):
                H_k = (H+H_add).copy()
                if not((aRy==0).all()):
                    H_k[diagonal(m,k=m-2*Ny*Nz)] = (-1j*aRy_kx)*np.exp(-1j*k_vec[i]*Nx)
                    H_k[diagonal(m,k=-m+2*Ny*Nz)] = (+1j*aRy_kx)*np.exp(1j*k_vec[i]*Nx)
                if not((aRz==0).all()):
                    H_k[diagonal(m,k=m-2*Ny*Nz-1,step=2,init=1)] = (-aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                    H_k[diagonal(m,k=-m+2*Ny*Nz+1,step=2,init=1)] = (-aRz_kx)*np.exp(1j*k_vec[i]*Nx)
                    H_k[diagonal(m,k=m+1-2*Ny*Nz,step=2)] = (aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                    H_k[diagonal(m,k=-m-1+2*Ny*Nz,step=2)] = (aRz_kx)*np.exp(1j*k_vec[i]*Nx)
                
                if np.isscalar(section) and section=='rectangular':
                    E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='SA',tol=1e-5)
                    E[:,i], U[:,:,i]=order_eig(E[:,i], U[:,:,i],sparse='no',BdG='no')

                else:
                    H_k=H_rec2shape(H_k,section,N,dis,BdG='no',output='H',m=m_hex)
                    E[:,i], U_hex = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_k),k = n_eig, which='SA',tol=1e-5)
                    E[:,i], U_hex =order_eig(E[:,i], U_hex,sparse='yes',BdG='no')
                    U[:,:,i]=U_shape2rec(U_hex,section,N,dis,BdG='no')

                            
    return (E), (U)



#%%    
def LO_3D_solver_MO(H,N,dis,
               n_eig,n_orb,Nxp=None,
               mu=0,aR=0,d=0,
               sparse='no',section='rectangular',BdG='yes'):

    """
    3D Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a 3D Lutchy-Oreg chain using the Benjamin D. Woods
    mehtod.
    
    Parameters
    ----------
        H_2D: arr
            A 1D array whose elements H_2D[i] are 2D arrays describing the
            cross-section Hamiltonian at the position x[i] of the wire. This is
            built with Lutchyn_builder.
            
        H_3D: arr
            The 3D Hamiltonian which includes the orbital-coupled terms. This
            is built with Lutchyn_builder.
            
        N: arr
            Number of sites in each direction.
            
        dis: int or int
            Distance (in nm) between sites. 
        
        mu: float or arr
            On-site chemical potential. If it is float, the chemical potential
            is the same in every site, while if it is a 3D array, it is the
            on-site chemical potential.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, the same constant aR is added in the z direction
            in each site and in every diagonalization step.
            -If aR is a 1D array of length=3, each element of the array is the
            (constant) Rashba coupling in each direction, which is added in 
            every diagonalization step.
            -If aR is a 3D array (3 x (N)), each element of the array aR[i] is
            the Rashba coupling in each direction, whose matrix alements are
            the on-site Rashba couplings.
            
        d: float or arr
            On-site supercondcuting pairing amplitude. If it is float, the 
            pairing is the same in every site, while if it is a 3D array,
            it is the on-site pairing.
            
        
        n_eig: int
            If sparse=="yes", n_eig is the number of eigenvalues you want to
            obtain. If BdG=='yes', these eigenvalues are obtained around zero
            energy, whil if BdG=='no' these eigenvalues correspond to the
            lowest-energy eigenstates. This can be changed with the near option.
            
        n_orb: int
            Number of molecular orbitals to include in the projection.
        
        Nxp: int
            Number of points to compute the molecular orbitals of the H_2D. For
            the remaining (N[0]-Nxp) slices, it is considered that the 
            molecular orbitals corresponding to the first (N[0]-Nxp)/2 slices
            are the same than for the slice N[Nxp]. Similarly, it is considered
            that for the last (N[0]-Nxp)/2 slices, the molecular orbitals are
            the same than that of N[N[0]-Nxp].
            
        sparse: {"yes","no"}
            Sparsety of the 2D Hamilonain. "yes" solves the Hamiltonian looking 
            for only n_eig eigenvalues, while "no" finds all.
            
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.
            
        BdG: {"yes","no"}
            If BdG is "yes", it is solved the Hamiltonian in the Bogoliubov-de
            Gennes formalism.            
            
            
    Returns
    -------
        E_3D: arr (n_eig)
            Eigevalues (energies), ordered from smaller to larger.
            
        U_1D: arr ((2 x N[0]) x n_eig)*(BdG=='yes') + ((N[0]) x n_eig)*(BdG=='no')
            Eigenvectors of the effective 1D problem.
            
        U_2D: arr ((2 x N[1] x N[2] x N[0]) x (n_orb x N[0]))
            Eigenvectors of the 2D slices in each point along the wire.
    """
    
    
    #Obtain dimensions:
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
        
    if not(Nxp==None or Nxp==N[0]):
        N_dif=np.int((Nx-Nxp)/2)
    else:
        Nxp, N_dif = Nx, 0    

    m = int(2 * Nx * Ny * Nz)
    
    #Make sure that the onsite parameters are arrays:    
    if BdG=='no':
        H_2D,H_3D=H
    elif BdG=='yes':
        H_2D,H_3D,H_SC=H
    
    
    if np.isscalar(mu):
        mu = mu * np.ones((Nx,Ny,Nz))
    else:
        if len(mu[:,0,0])<Nx and len(mu[:,0,0])==Nxp:
            mu_temp=np.zeros((Nx,Ny,Nz))
            for i in range(Nx):
                if i<=N_dif:
                    mu_temp[i,:,:]=mu[0,:,:]
                elif i>=(Nx-N_dif):
                    mu_temp[i,:,:]=mu[-1,:,:]
                else:
                    mu_temp[i,:,:]=mu[i-N_dif,:,:]
            mu=mu_temp
    
    if not(isinstance(aR,str) or isinstance(aR,dict)):
        if np.ndim(aR)==0:
            aRx=np.zeros((Nx,Ny,Nz))
            aRy=np.zeros((Nx,Ny,Nz))
            aRz=aR*np.ones((Nx,Ny,Nz))
        elif np.ndim(aR)==1:
            if len(aR)==3:
                aRx=aR[0]*np.ones((Nx,Ny,Nz))
                aRy=aR[1]*np.ones((Nx,Ny,Nz))
                aRz=aR[2]*np.ones((Nx,Ny,Nz))
            else:
                aRx=np.zeros((Nx,Ny,Nz))
                aRy=np.zeros((Nx,Ny,Nz))
                aRz=aR*np.ones((Nx,Ny,Nz))
        else:
            aRx=aR[0]
            aRy=aR[1]
            aRz=aR[2]
    else:
        aRx=np.zeros((Nx,Ny,Nz))
        aRy=np.zeros((Nx,Ny,Nz))
        aRz=np.zeros((Nx,Ny,Nz))
        
    if np.isscalar(d) and not(d==0) and BdG=='yes':
        d = d * np.ones((Nx,Ny,Nz))

    #Add new value:
    if not(isinstance(aR,str) or isinstance(aR,dict)) and not((aRy==0).all() and (aRz==0).all()):
        aRy_kx, aRz_kx = np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
        aRy_kx[1::2] = -aRy_kx[::2]
        H_3D[diagonal(m,k=2*Ny*Nz)] += -1j*aRy_kx
        H_3D[diagonal(m,k=-2*Ny*Nz)] += +1j*aRy_kx
        H_3D[diagonal(m,k=2*Ny*Nz-1,step=2,init=1)] += -aRz_kx
        H_3D[diagonal(m,k=-2*Ny*Nz+1,step=2,init=1)] += -aRz_kx
        H_3D[diagonal(m,k=1+2*Ny*Nz,step=2)] += aRz_kx
        H_3D[diagonal(m,k=-1-2*Ny*Nz,step=2)] += aRz_kx
    
    if not(np.isscalar(d)) and BdG=='yes':
        d=d.flatten()
        H_SC[diagonal(m,k=1,step=2)] += -np.conj(d)
        H_SC[diagonal(m,k=-1,step=2)] += np.conj(d)

    #Obtain the orbital basis:
    E_2D, U_2D = np.zeros(n_orb * Nx), scipy.sparse.dok_matrix((m,int(n_orb * Nx)),dtype=complex)
    for i in range(Nx):
        if i<N_dif:
            continue
        elif (i>=N_dif) and (i<=(Nx-N_dif)):
            if sparse=='no':
                E_temp, U_temp = LO_2D_solver_NoSC(H_2D[i-N_dif].todense(),N[1::],dis[1::],mu=mu[i,:,:],B=0,aR=np.array([aRx[i,:,:],aRy[i,:,:],aRz[i,:,:]]),space='position',sparse='no',section=section)
                E_2D[i*n_orb:(i+1)*n_orb], U_2D[2 * Ny * Nz * i:2 * Ny * Nz * (i+1),i*n_orb:(i+1)*n_orb] = order_eig(E_temp[0:n_orb],U_temp[:,0:n_orb],sparse='no',BdG='no')
            elif sparse=='yes':
                E_temp, U_temp = LO_2D_solver_NoSC(H_2D[i-N_dif],N[1::],dis[1::],mu=mu[i,:,:],B=0,aR=np.array([aRx[i,:,:],aRy[i,:,:],aRz[i,:,:]]),space='position',sparse='yes',n_eig=n_orb,section=section)
                E_2D[i*n_orb:(i+1)*n_orb], U_2D[2 * Ny * Nz * i:2 * Ny * Nz * (i+1),i*n_orb:(i+1)*n_orb] = order_eig(E_temp,U_temp,BdG='no')
            if i==N_dif:
                for j in range(N_dif):
                    E_2D[j*n_orb:(j+1)*n_orb], U_2D[2 * Ny * Nz * j:2 * Ny * Nz * (j+1),j*n_orb:(j+1)*n_orb] = E_2D[N_dif*n_orb:(N_dif+1)*n_orb], U_2D[2 * Ny * Nz * N_dif:2 * Ny * Nz * (N_dif+1),N_dif*n_orb:(N_dif+1)*n_orb]
        elif i>(Nx-N_dif):
            E_2D[i*n_orb:(i+1)*n_orb], U_2D[2 * Ny * Nz * i:2 * Ny * Nz * (i+1),i*n_orb:(i+1)*n_orb] = E_2D[(Nx-N_dif)*n_orb:((Nx-N_dif)+1)*n_orb], U_2D[2 * Ny * Nz * (Nx-N_dif):2 * Ny * Nz * ((Nx-N_dif)+1),(Nx-N_dif)*n_orb:((Nx-N_dif)+1)*n_orb]


    #Obtain the effective multiorbital 1D Hamiltonian:
    H_1D=(U_2D.transpose().conjugate()).dot(H_3D.dot(U_2D))+scipy.sparse.diags(E_2D)

    #Include the SC:    
    if BdG=='yes':
        H_1D_SC=((U_2D.transpose().conjugate()).dot(H_SC.dot(U_2D.conjugate())))
        H_1D=scipy.sparse.vstack([scipy.sparse.hstack([H_1D,H_1D_SC]),scipy.sparse.hstack([np.transpose(np.conj(H_1D_SC)),-np.conj(H_1D)])])
    
    #Diagonalize the effective 1D Hamiltonian:      
    if BdG=='no':
        E_3D,U_1D = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_1D),k = n_eig, which='SA',tol=1e-5)
    else:
        E_3D,U_1D = scipy.sparse.linalg.eigsh(scipy.sparse.csc_matrix(H_1D),k = n_eig,sigma=0, which='LM',tol=1e-5)

    #Project the eigenvector into the original basis:
    if BdG=='yes':
        zeros=scipy.sparse.dok_matrix((m,int(n_orb * Nx)),dtype=complex)
        U_2D_SC=scipy.sparse.vstack([scipy.sparse.hstack([U_2D,zeros]),scipy.sparse.hstack([zeros,np.conj(U_2D)])])
        U_3D=((U_2D_SC.dot(scipy.sparse.csc_matrix(U_1D))).todense()).A
        E_3D,U_3D=order_eig(E_3D,U_3D,sparse='yes',BdG='no')
    elif BdG=='no':
        U_3D=((U_2D.dot(scipy.sparse.csc_matrix(U_1D))).todense()).A
        E_3D,U_3D=order_eig(E_3D,U_3D,sparse='yes',BdG='yes')
        
    return (E_3D,U_3D)

