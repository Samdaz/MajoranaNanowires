
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
              "H_class/Lutchyn_Oreg/builders" submodule
                      
This sub-package builds Lutchyn-Oreg Hamiltonians for nanowires. Please, visit
http://www.samdaz/MajoranaNanowires.com for more details.

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

from MajoranaNanowires.Functions import order_eig, length, diagonal, H_rectangular2hexagonal, U_hexagonal2rectangular, concatenate




#%%
def LO_1D_builder(N,dis,m_eff,mu,B,aR,d, space='position', k_vec=np.nan ,sparse='no'):
    
    """
    1D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 1D
    Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        N: int or arr
            Number of sites.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is an array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is an array, each element is the on-site
            chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            an array with the on-site Rashba couplings in the direction i.
            
        d: float or arr
            Superconductor paring amplitud.
            -If d is a float, d is the Rashba coupling along the y-direction,
            with the same value in every site.
            -If d is an array, each element of the array is the on-site
            superconducting paring amplitud
            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones(N)
        
    if np.isscalar(mu):
        mu = mu * np.ones(N)
        
    if np.isscalar(B):
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
    
    if space=='momentum':
        n_k=len(k_vec)
    
    #Obtain the hopping and on-site energies:
    t=cons.hbar**2/(2*m_eff*cons.m_e*(dis*1e-9)**2)/cons.e*1e3
    e = 2 * t - mu

    ##Build the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(4 * N), int(4 * N)),dtype=complex)
    elif sparse=='yes':
        H=scipy.sparse.dok_matrix((int(4*N),int(4*N)),dtype=complex)
           
    t, aRy, Bz = np.repeat(t,2), np.repeat(aRy,2), np.repeat(Bz,2)
    Bz[1::2], aRy[1::2] = -Bz[::2], -aRy[::2]
    
    for i in range(2):
        H[diagonal(2*N*(i+1),init=2*N*i,k=1,step=2)], H[diagonal(2*N*(i+1),init=2*N*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
        H[diagonal(2*N*(i+1),init=2*N*i)] = (-1)**i*(Bz+np.repeat(e,2))
        
        H[diagonal(2*N*(i+1),init=2*N*i,k=-2)] = -1*(-1)**i*t[2::]+1j*aRy[2::]
        H[diagonal(2*N*(i+1),init=2*N*i,k=2)] = -1*(-1)**i*t[2::]-1j*aRy[2::]
                
        H[diagonal(2*N*(i+1),k=1,step=2,init=1+2*N*i)] += -1*(-1)**i*aRz[1::]
        H[diagonal(2*N*(i+1),k=-1,step=2,init=1+2*N*i)] += -1*(-1)**i*aRz[1::]
        H[diagonal(2*N*(i+1),init=2*N*i,k=3,step=2)] += (-1)**i*aRz[1::]
        H[diagonal(2*N*(i+1),init=2*N*i,k=-3,step=2)] += (-1)**i*aRz[1::]
        
    H[diagonal(4*N,k=2*N+1,step=2)], H[diagonal(4*N,k=-2*N-1,step=2)] = -np.conj(d), -d
    H[diagonal(4*N,k=2*N-1,step=2,init=1)], H[diagonal(4*N,k=-2*N+1,step=2,init=1)] = np.conj(d), d
    
    
    #Build it in momentum space if required:
    if space=='momentum':
        if sparse=='no':
            H_k = np.zeros((int(4 * N), int(4 * N), int(n_k)),dtype=complex)
            for i in range(n_k):
                H_k[:,:,i]=H
                H_k[2 * (N - 1):2 * (N - 1) + 2, 0: 2,i] += np.array([[-t[2]-1j*aRy[2], aRz[1]], [-aRz[1], -t[2]+1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
                H_k[2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N, 2*N: 2+2*N,i] += -np.array([[-t[2]+1j*aRy[2], aRz[1]], [-aRz[1], -t[2]-1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)
                
                H_k[0: 2, 2 * (N - 1):2 * (N - 1) + 2,i] += np.array([[-t[2]+1j*aRy[2], -aRz[1]], [aRz[1], -t[2]-1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)
                H_k[2*N: 2+2*N, 2 * (N - 1)+2*N:2 * (N - 1) + 2+2*N,i] += -np.array([[-t[2]-1j*aRy[2], -aRz[1]], [aRz[1], -t[2]+1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
            return (H_k)
        
        elif sparse=='yes':
            return(H)
    
    else:    
        return (H)
    
    
#%%
def LO_1D_builder_NoSC(N,dis,m_eff,mu,B,aR, space='position', k_vec=np.nan ,sparse='no'):
    
    """
    1D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 1D
    Lutchy-Oreg chain without superconductivity.
    
    Parameters
    ----------
        N: int or arr
            Number of sites.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is an array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is an array, each element is the on-site
            chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            an array with the on-site Rashba couplings in the direction i. 
            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones(N)
        
    if np.isscalar(mu):
        mu = mu * np.ones(N)
        
    if np.isscalar(B):
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
        
    if space=='momentum':
        n_k=len(k_vec)
        
    #Obtain the hopping and the on-site energies:
    t=cons.hbar**2/(2*m_eff*cons.m_e*(dis*1e-9)**2)/cons.e*1e3
    e = 2 * t - mu

    ##Build the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(2 * N), int(2 * N)),dtype=complex)
    elif sparse=='yes':
        H=scipy.sparse.dok_matrix((int(2*N),int(2*N)),dtype=complex)
                
    Bz,Bx,By=np.repeat(Bz,2),np.repeat(Bx,2), 1j*np.repeat(By,2)
    Bx[1::2], By[1::2], Bz[1::2] = 0, 0, -Bz[::2]
    H[diagonal(2*N,k=1)], H[diagonal(2*N,k=-1)] = Bx[:-1]-By[:-1], Bx[:-1]+By[:-1]
    H[diagonal(2*N)]=Bz+np.repeat(e,2)
    
    t=-np.repeat(t,2)
    aRy=np.repeat(aRy,2)
    aRy[1::2]= -aRy[::2]
    H[diagonal(2*N,k=-2)], H[diagonal(2*N,k=2)] = t[2::]+1j*aRy[2::], t[2::]-1j*aRy[2::]
    H[diagonal(2*N,k=1,step=2,init=1)] += -aRz[1::]
    H[diagonal(2*N,k=-1,step=2,init=1)] += -aRz[1::]
    H[diagonal(2*N,k=3,step=2)] += aRz[1::]
    H[diagonal(2*N,k=-3,step=2)] += aRz[1::]
    
    #Build it in momentum space if required:            
    if space=='momentum':
        if sparse=='no':
            H_k = np.zeros((int(2 * N), int(2 * N), int(n_k)),dtype=complex)
            for i in range(n_k):
                H_k[:,:,i]=H
                H_k[2 * (N - 1):2 * (N - 1) + 2, 0: 2,i] += np.array([[-t[2]-1j*aRy[2], aRz[1]], [-aRz[1], -t[2]+1j*aRy[2]]])*np.exp(-1j*k_vec[i]*N)
                H_k[0: 2, 2 * (N - 1):2 * (N - 1) + 2,i] += np.array([[-t[2]+1j*aRy[2], -aRz[1]], [aRz[1], -t[2]-1j*aRy[2]]])*np.exp(1j*k_vec[i]*N)
            return (H_k)    
        
        elif sparse=='yes':
            return (H)
    
    else:    
        return (H)




#%%
def LO_2D_builder(N,dis,m_eff,mu,B,aR, d, space='position', k_vec=np.nan ,sparse='no'):
  
    """
    2D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 2D
    Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is a 2D array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is a 2D array, each element is the 
            on-site chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            a 2D array with the on-site Rashba couplings in the direction i. 
            
        d: float or arr
            Superconductor paring amplitud.
            -If d is a float, d is the Rashba coupling along the y-direction,
            with the same value in every site.
            -If d is a 2D array, each element of the array is the on-site
            superconducting paring amplitud
            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    #Obtain the dimensions:
    Ny, Nz = N[0], N[1]
    
    if np.ndim(dis)==0:
        dis_y, dis_z = dis, dis
    else: 
        dis_y, dis_z = dis[0], dis[1]

    m = 4 * Ny * Nz 
        
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones((Ny,Nz))
        
    if np.isscalar(mu):
        mu = mu * np.ones((Ny,Nz))
        
    if np.isscalar(B):
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
    
    if np.isscalar(d):
        d = d * np.ones(N)
        
        
    #Obtain the eigenenergies:
    ty=cons.hbar**2/(2*(m_eff[1::,:]+m_eff[:-1,:])/2*cons.m_e*(dis_y*1e-9)**2)/cons.e*1e3
    tz=cons.hbar**2/(2*(m_eff[:,1::]+m_eff[:,:-1])/2*cons.m_e*(dis_z*1e-9)**2)/cons.e*1e3
    e = - mu
    e += np.append(2*ty[0,:].reshape(1,Nz),np.append(ty[1::,:]+ty[:-1,:],2*ty[-1,:].reshape(1,Nz),axis=0),axis=0)
    e += np.append(2*tz[:,0].reshape(Ny,1),np.append(tz[:,1::]+tz[:,:-1],2*tz[:,-1].reshape(Ny,1),axis=1),axis=1)


    #Build the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(m), int(m)),dtype=complex)
    elif sparse=='yes':
        H = scipy.sparse.dok_matrix((int(m),int(m)),dtype=complex)
        
    e,d,Bx,By,Bz=e.flatten(),d.flatten(),Bx.flatten(),By.flatten(),Bz.flatten()
    Bz=np.repeat(Bz,2)
    Bz[1::2] = -Bz[::2]
    ty, aRx_ky, aRz_ky = np.repeat(ty.flatten(),2), np.repeat(((aRx[1::,:]+aRx[:-1,:])/(4*dis_y)).flatten(),2), ((aRz[1::,:]+aRz[:-1,:])/(4*dis_y)).flatten()
    tz, aRx_kz, aRy_kz = np.repeat(tz.flatten(),2), ((aRx[:,1::]+aRx[:,:-1])/(4*dis_z)).flatten(), ((aRy[:,1::]+aRy[:,:-1])/(4*dis_z)).flatten()
    aRx_ky[1::2] = -aRx_ky[::2] 
    tz, aRx_kz, aRy_kz=np.insert(tz,np.repeat(np.arange(2*(Nz-1),2*(Nz-1)*Ny,2*(Nz-1)),2),np.zeros(2*(Ny-1))), np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1)))

    
    for i in range(2):
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1,step=2)], H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] = (-1)**(i)*(np.repeat(e,2) + Bz)
        
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Nz)] = -1*(-1)**(i)*ty+1j*aRx_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Nz)] = -1*(-1)**(i)*ty-1j*aRx_ky
        H[diagonal(int(m/2)*(i+1),k=2*Nz-1,step=2,init=1+int(m/2)*i)] += -1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),k=-2*Nz+1,step=2,init=1+int(m/2)*i)] += 1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Nz,step=2)] += -1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Nz,step=2)] += 1j*aRz_ky
        
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2)] = -1*(-1)**(i)*tz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2)] = -1*(-1)**(i)*tz
        H[diagonal(int(m/2)*(i+1),k=1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz+1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),k=-1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz-1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=3,step=2)] += -1*(-1)**(i)*aRx_kz+1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-3,step=2)] += -1*(-1)**(i)*aRx_kz-1j*aRy_kz
    
    H[diagonal(m,k=int(m/2)+1,step=2)], H[diagonal(m,k=-int(m/2)-1,step=2)] = -np.conj(d), -d
    H[diagonal(m,k=int(m/2)-1,step=2,init=1)], H[diagonal(m,k=-int(m/2)+1,step=2,init=1)] = np.conj(d), d

    return (H)


#%%
def LO_2D_builder_NoSC(N,dis,m_eff,mu,B,aR, space='position', k_vec=np.nan ,sparse='no'):

    """
    2D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 2D
    Lutchy-Oreg chain without superconductivity.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is a 2D array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is a 2D array, each element is the 
            on-site chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            a 2D array with the on-site Rashba couplings in the direction i. 
            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    #Obtain the dimensions:
    Ny, Nz = N[0], N[1]
    
    if np.ndim(dis)==0:
        dis_y, dis_z = dis, dis
    else: 
        dis_y, dis_z = dis[0], dis[1]

    m = 2 * Ny * Nz 
        
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones((Ny,Nz))
        
    if np.isscalar(mu):
        mu = mu * np.ones((Ny,Nz))
        
    if np.isscalar(B):
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
        
        
    #Obtain the eigenenergies:
    ty=cons.hbar**2/(2*(m_eff[1::,:]+m_eff[:-1,:])/2*cons.m_e*(dis_y*1e-9)**2)/cons.e*1e3
    tz=cons.hbar**2/(2*(m_eff[:,1::]+m_eff[:,:-1])/2*cons.m_e*(dis_z*1e-9)**2)/cons.e*1e3
    e = - mu
    e += np.append(2*ty[0,:].reshape(1,Nz),np.append(ty[1::,:]+ty[:-1,:],2*ty[-1,:].reshape(1,Nz),axis=0),axis=0)
    e += np.append(2*tz[:,0].reshape(Ny,1),np.append(tz[:,1::]+tz[:,:-1],2*tz[:,-1].reshape(Ny,1),axis=1),axis=1)

    #Build the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(m), int(m)),dtype=complex)
    elif sparse=='yes':
        H = scipy.sparse.dok_matrix((int(m),int(m)),dtype=complex)
    

    e,Bx,By,Bz=e.flatten(),Bx.flatten(),By.flatten(),Bz.flatten()
    Bz=np.repeat(Bz,2)
    Bz[1::2] = -Bz[::2]
    ty, aRx_ky, aRz_ky = np.repeat(ty.flatten(),2), np.repeat(((aRx[1::,:]+aRx[:-1,:])/(4*dis_y)).flatten(),2), ((aRz[1::,:]+aRz[:-1,:])/(4*dis_y)).flatten()
    tz, aRx_kz, aRy_kz = np.repeat(tz.flatten(),2), ((aRx[:,1::]+aRx[:,:-1])/(4*dis_z)).flatten(), ((aRy[:,1::]+aRy[:,:-1])/(4*dis_z)).flatten()
    aRx_ky[1::2] = -aRx_ky[::2] 
    
    H[diagonal(m,k=1,step=2)], H[diagonal(m,k=-1,step=2)] = Bx-1j*By, Bx+1j*By
    H[diagonal(m)] = np.repeat(e,2) + Bz
    
    H[diagonal(m,k=2*Nz)] = -ty+1j*aRx_ky
    H[diagonal(m,k=-2*Nz)] = -ty-1j*aRx_ky
    H[diagonal(m,k=2*Nz-1,step=2,init=1)] += -1j*aRz_ky
    H[diagonal(m,k=-2*Nz+1,step=2,init=1)] += 1j*aRz_ky
    H[diagonal(m,k=1+2*Nz,step=2)] += -1j*aRz_ky
    H[diagonal(m,k=-1-2*Nz,step=2)] += 1j*aRz_ky
    
    tz, aRx_kz, aRy_kz=np.insert(tz,np.repeat(np.arange(2*(Nz-1),2*(Nz-1)*Ny,2*(Nz-1)),2),np.zeros(2*(Ny-1))), np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny,(Nz-1)),np.zeros((Ny-1)))
    H[diagonal(m,k=2)] = -tz
    H[diagonal(m,k=-2)] = -tz
    H[diagonal(m,k=1,step=2,init=1)] += aRx_kz+1j*aRy_kz
    H[diagonal(m,k=-1,step=2,init=1)] += aRx_kz-1j*aRy_kz
    H[diagonal(m,k=3,step=2)] += -aRx_kz+1j*aRy_kz
    H[diagonal(m,k=-3,step=2)] += -aRx_kz-1j*aRy_kz
        
    return (H)





#%%
def LO_3D_builder(N,dis,m_eff,mu,B,aR,d, space='position', k_vec=np.nan ,sparse='yes'):

    
    """
    3D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 3D
    Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is a 3D array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is a 3D array, each element is the 
            on-site chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            a 3D array with the on-site Rashba couplings in the direction i. 
            
        d: float or arr
            Superconductor paring amplitud.
            -If d is a float, d is the Rashba coupling along the y-direction,
            with the same value in every site.
            -If d is a 3D array, each element of the array is the on-site
            superconducting paring amplitud

            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    
    #Obtain the dimensions:
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
        
    m = 4 * Nx * Ny * Nz 
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones((Nx,Ny,Nz))
    
    if np.isscalar(mu):
        mu = mu * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(B):
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
        
    if np.isscalar(d):
        d = d * np.ones((Nx,Ny,Nz))
        
    if space=='momentum':
        n_k=len(k_vec)
        
        
    #Obtain the hoppings and the on-site energies:
    tx=cons.hbar**2/(2*(m_eff[1::,:,:]+m_eff[:-1,:,:])/2*cons.m_e*(dis_x*1e-9)**2)/cons.e*1e3
    ty=cons.hbar**2/(2*(m_eff[:,1::,:]+m_eff[:,:-1,:])/2*cons.m_e*(dis_y*1e-9)**2)/cons.e*1e3
    tz=cons.hbar**2/(2*(m_eff[:,:,1::]+m_eff[:,:,:-1])/2*cons.m_e*(dis_z*1e-9)**2)/cons.e*1e3
    e = - mu
    e += np.append(2*tx[0,:,:].reshape(1,Ny,Nz),np.append(tx[1::,:,:]+tx[:-1,:,:],2*tx[-1,:,:].reshape(1,Ny,Nz),axis=0),axis=0)
    e += np.append(2*ty[:,0,:].reshape(Nx,1,Nz),np.append(ty[:,1::,:]+ty[:,:-1,:],2*ty[:,-1,:].reshape(Nx,1,Nz),axis=1),axis=1)
    e += np.append(2*tz[:,:,0].reshape(Nx,Ny,1),np.append(tz[:,:,1::]+tz[:,:,:-1],2*tz[:,:,-1].reshape(Nx,Ny,1),axis=2),axis=2)

    
    #Built the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(m), int(m)),dtype=complex)
    elif sparse=='yes':
        H = scipy.sparse.dok_matrix((int(m),int(m)),dtype=complex)
        
    e,d,Bx,By,Bz=e.flatten(),d.flatten(),Bx.flatten(),By.flatten(),Bz.flatten()
    Bz=np.repeat(Bz,2)
    Bz[1::2] = -Bz[::2]
    tx, aRy_kx, aRz_kx = np.repeat(tx.flatten(),2), np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
    ty, aRx_ky, aRz_ky = np.repeat(ty.flatten(),2), np.repeat(((aRx[:,1::,:]+aRx[:,:-1,:])/(4*dis_y)).flatten(),2), ((aRz[:,1::,:]+aRz[:,:-1,:])/(4*dis_y)).flatten()
    tz, aRx_kz, aRy_kz = np.repeat(tz.flatten(),2), ((aRx[:,:,1::]+aRx[:,:,:-1])/(4*dis_z)).flatten(), ((aRy[:,:,1::]+aRy[:,:,:-1])/(4*dis_z)).flatten()
    aRy_kx[1::2], aRx_ky[1::2] = -aRy_kx[::2], -aRx_ky[::2] 
    ty, aRx_ky, aRz_ky = np.insert(ty,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))), np.insert(aRx_ky,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))),np.insert(aRz_ky,np.repeat(np.arange((Nz*Ny-Nz),(Ny*Nz-Nz)*Nx,(Ny*Nz-Nz)),Nz),np.zeros(Nz*(Nx-1)))
    tz, aRx_kz, aRy_kz=np.insert(tz,np.repeat(np.arange(2*(Nz-1),2*(Nz-1)*Ny*Nx,2*(Nz-1)),2),np.zeros(2*Nx*(Ny-1)+2*(Nx-1))), np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1)))

    for i in range(2):
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1,step=2)], H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1,step=2)] = (-1)**(i)*Bx-1j*By, (-1)**(i)*Bx+1j*By
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i)] = (-1)**(i)*np.repeat(e,2) + (-1)**(i)*Bz
    
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Ny*Nz)] = -1*(-1)**(i)*tx-1j*aRy_kx
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Ny*Nz)] = -1*(-1)**(i)*tx+1j*aRy_kx
        H[diagonal(int(m/2)*(i+1),k=2*Ny*Nz-1,step=2,init=1+int(m/2)*i)] += -1*(-1)**(i)*aRz_kx
        H[diagonal(int(m/2)*(i+1),k=-2*Ny*Nz+1,step=2,init=1+int(m/2)*i)] += -1*(-1)**(i)*aRz_kx
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Ny*Nz,step=2)] += (-1)**(i)*aRz_kx
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Ny*Nz,step=2)] += (-1)**(i)*aRz_kx
        
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2*Nz)] = -1*(-1)**(i)*ty+1j*aRx_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2*Nz)] = -1*(-1)**(i)*ty-1j*aRx_ky
        H[diagonal(int(m/2)*(i+1),k=2*Nz-1,step=2,init=1+int(m/2)*i)] += -1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),k=-2*Nz+1,step=2,init=1+int(m/2)*i)] += 1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=1+2*Nz,step=2)] += -1j*aRz_ky
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-1-2*Nz,step=2)] += 1j*aRz_ky
        
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=2)] = -1*(-1)**(i)*tz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-2)] = -1*(-1)**(i)*tz
        H[diagonal(int(m/2)*(i+1),k=1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz+1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),k=-1,step=2,init=1+int(m/2)*i)] += (-1)**(i)*aRx_kz-1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=3,step=2)] += -1*(-1)**(i)*aRx_kz+1j*aRy_kz
        H[diagonal(int(m/2)*(i+1),init=int(m/2)*i,k=-3,step=2)] += -1*(-1)**(i)*aRx_kz-1j*aRy_kz
    
    H[diagonal(m,k=int(m/2)+1,step=2)], H[diagonal(m,k=-int(m/2)-1,step=2)] = -np.conj(d), -d
    H[diagonal(m,k=int(m/2)-1,step=2,init=1)], H[diagonal(m,k=-int(m/2)+1,step=2,init=1)] = np.conj(d), d
        
        
    #Build it in momentum space if required:  
    if space=='momentum':
        if sparse=='no':
            H_k = np.zeros((int(m), int(m), int(n_k)),dtype=complex)
            for i in range(n_k):
                H_k[:,:,i] = H
                for j in range(2):
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m-2*Ny*Nz),i] = (-1*(-1)**(j)*tx-1j*aRy_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx)
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m+2*Ny*Nz),i] = (-1*(-1)**(j)*tx+1j*aRy_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx)
                    H_k[diagonal(int(m/2)*(j+1),k=m-2*Ny*Nz-1,step=2,init=1+int(m/2)*j),i] += (-1)**(j)*(-aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx)
                    H_k[diagonal(int(m/2)*(j+1),k=-m+2*Ny*Nz+1,step=2,init=1+int(m/2)*j),i] += (-1)**(j)*(-aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx)
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=m+1-2*Ny*Nz,step=2),i] += (-1)**(j)*(aRz_kx)*np.exp(-1j*(-1)**(i)*k_vec[i]*Nx)
                    H_k[diagonal(int(m/2)*(j+1),init=int(m/2)*j,k=-m-1+2*Ny*Nz,step=2),i] += (-1)**(j)*(aRz_kx)*np.exp(1j*(-1)**(i)*k_vec[i]*Nx)
            return (H_k)
        
        elif sparse=='yes':
            return(H)
    
    else:
        return (H)


#%%
def LO_3D_builder_NoSC(N,dis,m_eff,mu,B,aR, space='position', k_vec=np.nan ,sparse='no'):
    
    """
    3D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a 3D
    Lutchy-Oreg chain with superconductivity.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is a 3D array, each element is the on-site
            effective mass.
        
        mu: float or arr
            Chemical potential. If it is a 3D array, each element is the 
            on-site chemical potential
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            a 3D array with the on-site Rashba couplings in the direction i. 
            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    
    #Obtain the dimensions:
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
        
    m = 2 * Nx * Ny * Nz 
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(mu):
        mu = mu * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(B):
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
        
    if space=='momentum':
        n_k=len(k_vec)
        
    #Obtain the hoppings and the on-site energies:
    tx=cons.hbar**2/(2*(m_eff[1::,:,:]+m_eff[:-1,:,:])/2*cons.m_e*(dis_x*1e-9)**2)/cons.e*1e3
    ty=cons.hbar**2/(2*(m_eff[:,1::,:]+m_eff[:,:-1,:])/2*cons.m_e*(dis_y*1e-9)**2)/cons.e*1e3
    tz=cons.hbar**2/(2*(m_eff[:,:,1::]+m_eff[:,:,:-1])/2*cons.m_e*(dis_z*1e-9)**2)/cons.e*1e3
    e = - mu
    e += np.append(2*tx[0,:,:].reshape(1,Ny,Nz),np.append(tx[1::,:,:]+tx[:-1,:,:],2*tx[-1,:,:].reshape(1,Ny,Nz),axis=0),axis=0)
    e += np.append(2*ty[:,0,:].reshape(Nx,1,Nz),np.append(ty[:,1::,:]+ty[:,:-1,:],2*ty[:,-1,:].reshape(Nx,1,Nz),axis=1),axis=1)
    e += np.append(2*tz[:,:,0].reshape(Nx,Ny,1),np.append(tz[:,:,1::]+tz[:,:,:-1],2*tz[:,:,-1].reshape(Nx,Ny,1),axis=2),axis=2)

    
    #Built the Hamiltonian:
    if sparse=='no':
        H = np.zeros((int(m), int(m)),dtype=complex)
    elif sparse=='yes':
        H = scipy.sparse.dok_matrix((int(m),int(m)),dtype=complex)
        
    e,Bx,By,Bz=e.flatten(),Bx.flatten(),By.flatten(),Bz.flatten()
    Bz=np.repeat(Bz,2)
    Bz[1::2] = -Bz[::2]
    tx, aRy_kx, aRz_kx = np.repeat(tx.flatten(),2), np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
    ty, aRx_ky, aRz_ky = np.repeat(ty.flatten(),2), np.repeat(((aRx[:,1::,:]+aRx[:,:-1,:])/(4*dis_y)).flatten(),2), ((aRz[:,1::,:]+aRz[:,:-1,:])/(4*dis_y)).flatten()
    tz, aRx_kz, aRy_kz = np.repeat(tz.flatten(),2), ((aRx[:,:,1::]+aRx[:,:,:-1])/(4*dis_z)).flatten(), ((aRy[:,:,1::]+aRy[:,:,:-1])/(4*dis_z)).flatten()
    aRy_kx[1::2], aRx_ky[1::2] = -aRy_kx[::2], -aRx_ky[::2] 
    
    H[diagonal(m,k=1,step=2)], H[diagonal(m,k=-1,step=2)] = Bx-1j*By, Bx+1j*By
    H[diagonal(m)] = np.repeat(e,2) + Bz

    H[diagonal(m,k=2*Ny*Nz)] = -tx-1j*aRy_kx
    H[diagonal(m,k=-2*Ny*Nz)] = -tx+1j*aRy_kx
    H[diagonal(m,k=2*Ny*Nz-1,step=2,init=1)] += -aRz_kx
    H[diagonal(m,k=-2*Ny*Nz+1,step=2,init=1)] += -aRz_kx
    H[diagonal(m,k=1+2*Ny*Nz,step=2)] += aRz_kx
    H[diagonal(m,k=-1-2*Ny*Nz,step=2)] += aRz_kx
    
    ty, aRx_ky, aRz_ky = np.insert(ty,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))), np.insert(aRx_ky,np.repeat(np.arange(2*(Nz*Ny-Nz),2*(Ny*Nz-Nz)*Nx,2*(Ny*Nz-Nz)),2*Nz),np.zeros(2*Nz*(Nx-1))),np.insert(aRz_ky,np.repeat(np.arange((Nz*Ny-Nz),(Ny*Nz-Nz)*Nx,(Ny*Nz-Nz)),Nz),np.zeros(Nz*(Nx-1)))
    H[diagonal(m,k=2*Nz)] = -ty+1j*aRx_ky
    H[diagonal(m,k=-2*Nz)] = -ty-1j*aRx_ky
    H[diagonal(m,k=2*Nz-1,step=2,init=1)] += -1j*aRz_ky
    H[diagonal(m,k=-2*Nz+1,step=2,init=1)] += 1j*aRz_ky
    H[diagonal(m,k=1+2*Nz,step=2)] += -1j*aRz_ky
    H[diagonal(m,k=-1-2*Nz,step=2)] += 1j*aRz_ky
    
    tz, aRx_kz, aRy_kz=np.insert(tz,np.repeat(np.arange(2*(Nz-1),2*(Nz-1)*Ny*Nx,2*(Nz-1)),2),np.zeros(2*Nx*(Ny-1)+2*(Nx-1))), np.insert(aRx_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1))), np.insert(aRy_kz,np.arange((Nz-1),(Nz-1)*Ny*Nx,(Nz-1)),np.zeros(Nx*(Ny-1)+(Nx-1)))
    H[diagonal(m,k=2)] = -tz
    H[diagonal(m,k=-2)] = -tz
    H[diagonal(m,k=1,step=2,init=1)] += aRx_kz+1j*aRy_kz
    H[diagonal(m,k=-1,step=2,init=1)] += aRx_kz-1j*aRy_kz
    H[diagonal(m,k=3,step=2)] += -aRx_kz+1j*aRy_kz
    H[diagonal(m,k=-3,step=2)] += -aRx_kz-1j*aRy_kz
        
        
    #Build it in momentum space if required:  
    if space=='momentum':
        if sparse=='no':
            H_k = np.zeros((int(m), int(m), int(n_k)),dtype=complex)
            for i in range(n_k):
                H_k[:,:,i] = H
                H_k[diagonal(m,k=m-2*Ny*Nz),i] = (-tx-1j*aRy_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_k[diagonal(m,k=-m+2*Ny*Nz),i] = (-tx+1j*aRy_kx)*np.exp(1j*k_vec[i]*Nx)
                H_k[diagonal(m,k=m-2*Ny*Nz-1,step=2,init=1),i] += (-aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_k[diagonal(m,k=-m+2*Ny*Nz+1,step=2,init=1),i] += (-aRz_kx)*np.exp(1j*k_vec[i]*Nx)
                H_k[diagonal(m,k=m+1-2*Ny*Nz,step=2),i] += (aRz_kx)*np.exp(-1j*k_vec[i]*Nx)
                H_k[diagonal(m,k=-m-1+2*Ny*Nz,step=2),i] += (aRz_kx)*np.exp(1j*k_vec[i]*Nx)
            return (H_k)
        
        elif sparse=='yes':
            return(H)
    
    else:
        return (H)



#%%
def LO_3D_addOrbital(N,dis,m_eff,B,aR,sparse='yes'):

    """
    Add orbital effects to the 3D Lutchyn-Oreg Hamiltonian with 
    superconductivity. It builds the Hamiltonian with only the orbital effects.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        m_eff: int or arr
            Effective mass. If it is a 3D array, each element is the on-site
            effective mass.
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            a 3D array with the on-site Rashba couplings in the direction i.
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
           
            
    Returns
    -------
        H: arr
            Hamiltonian with only the orbital effects.
    """
    
    
    #Obtain the dimensions:
    Nx, Ny, Nz = N[0], N[1], N[2]
    
    if np.ndim(dis)==0:
        dis_x, dis_y, dis_z = dis, dis, dis
    else: 
        dis_x, dis_y, dis_z = dis[0], dis[1], dis[2]
    
    m = 4 * Nx * Ny * Nz 
    
    #Make sure that the onsite parameters are arrays:
    if np.isscalar(B):
        Bx=B
    else:
        Bx=B[0]
    
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

    y, z = np.linspace(-Ny*dis_y/2,Ny*dis_y/2,Ny), np.linspace(-Nz*dis_z/2,Nz*dis_z/2,Nz)

    #Obtain some constants:        
    f1,f2,f3 = cons.e**2*Bx/(8*cons.m_e*m_eff)/cons.e*1e3*(1e-9)**2, cons.hbar*cons.e*Bx/(2*m_eff*cons.m_e)/cons.e*1e3, Bx*cons.e/(2*cons.hbar)*(1e-9)**2

    #Dictionary of sites:
    dic=np.zeros((Nx,Ny,Nz,2),dtype=int)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                dic[i,j,k,:]=np.array([2*(k+(j+i*Ny)*Nz),2*(k+(j+i*Ny)*Nz)+1])
    
    #Built the Hamiltonian of orbital effects:
    if sparse=='no':
        H = np.zeros((int(m), int(m)),dtype=complex)
    elif sparse=='yes':
        H = scipy.sparse.dok_matrix((int(m),int(m)),dtype=complex)
    
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                H[np.ix_(dic[i,j,k,:],dic[i,j,k,:])] = np.array([[f1*(y[j]**2+z[k]**2)+f3*aRx[i,j,k]*z[k], -f3*(aRz[i,j,k]*z[k]+aRy[i,j,k]*y[j])-1j*f3*aRx[i,j,k]*y[j]], [-f3*(aRz[i,j,k]*z[k]+aRy[i,j,k]*y[j])+1j*f3*aRx[i,j,k]*y[j], f1*(y[j]**2+z[k]**2)-f3*aRx[i,j,k]*z[k]]])
                H[np.ix_(dic[i,j,k,:]+int(m/2),dic[i,j,k,:]+int(m/2))] = -np.array([[f1*(y[j]**2+z[k]**2)+f3*aRx[i,j,k]*z[k], -f3*(aRz[i,j,k]*z[k]+aRy[i,j,k]*y[j])+1j*f3*aRx[i,j,k]*y[j]], [-f3*(aRz[i,j,k]*z[k]+aRy[i,j,k]*y[j])-1j*f3*aRx[i,j,k]*y[j], f1*(y[j]**2+z[k]**2)-f3*aRx[i,j,k]*z[k]]])              
                                
                if (k>0):
                    H[np.ix_(dic[i,j,k-1,:],dic[i,j,k,:])]=np.array([[-1j*f2*y[j]/(2*dis_z),0], [0,-1j*f2*y[j]/(2*dis_z)]])
                    H[np.ix_(dic[i,j,k,:],dic[i,j,k-1,:])]=np.array([[1j*f2*y[j]/(2*dis_z),0], [0,1j*f2*y[j]/(2*dis_z)]])
                    
                    H[np.ix_(dic[i,j,k-1,:]+int(m/2),dic[i,j,k,:]+int(m/2))]=-np.array([[1j*f2*y[j]/(2*dis_z),0], [0,1j*f2*y[j]/(2*dis_z)]])
                    H[np.ix_(dic[i,j,k,:]+int(m/2),dic[i,j,k-1,:]+int(m/2))]=-np.array([[-1j*f2*y[j]/(2*dis_z),0], [0,-1j*f2*y[j]/(2*dis_z)]])
        
                if (j>0):
                    H[np.ix_(dic[i,j-1,k,:],dic[i,j,k,:])]=np.array([[1j*f2*z[k]/(2*dis_y),0], [0,1j*f2*z[k]/(2*dis_y)]])
                    H[np.ix_(dic[i,j,k,:],dic[i,j-1,k,:])]=np.array([[-1j*f2*z[k]/(2*dis_y),0], [0,-1j*f2*z[k]/(2*dis_y)]])
                    
                    H[np.ix_(dic[i,j-1,k,:]+int(m/2),dic[i,j,k,:]+int(m/2))]=-np.array([[-1j*f2*z[k]/(2*dis_y),0], [0,-1j*f2*z[k]/(2*dis_y)]])
                    H[np.ix_(dic[i,j,k,:]+int(m/2),dic[i,j-1,k,:]+int(m/2))]=-np.array([[1j*f2*z[k]/(2*dis_y),0], [0,1j*f2*z[k]/(2*dis_y)]])
            
    return (H)



#%%
def LO_3D_builder_MO(N,dis,m_eff,
                mu,B,aR,d=0,
                BdG='yes',
                Nxp=None):
    
    """
    3D Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a
    3D Lutchy-Oreg chain with the method of Benjamin D. Woods.
    
    Parameters
    ----------
        N: int or arr
            Number of sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites in each direction.
            
        m_eff: int or arr
            Effective mass. If it is a 3D array, each element is the effective
            mass on each site of the lattice.
        
        mu: float or arr
            Chemical potential. If it is a 3D array, each element is the
            chemical potential on each site of the lattice.
            
        B: float or arr
            Zeeman splitting. If it is an array, each element is the Zeeman
            splitting in each direction.
            
        aR: float or arr
            Rashba coupling.
            -If aR is a float, aR is the Rashba coupling along the z-direction,
            with the same value in every site.
            -If aR is a 1D array with length=3, each element of the array is
            the rashba coupling in each direction.
            -If aR is an array of arrays (3 x N), each element of aR[i] is 
            an array (1D, 2D or 3D) with the on-site Rashba couplings in the 
            direction i. 
            
        d: float or arr
            On-site supercondcuting pairing amplitude. If it is float, the 
            pairing is the same in every site, while if it is a 3D array,
            it is the on-site pairing.
            
        
        Nxp: int
            Number of points to compute the molecular orbitals of the H_2D. For
            the remaining (N[0]-Nxp) slices, it is considered that the 
            molecular orbitals corresponding to the first (N[0]-Nxp)/2 slices
            are the same than for the slice N[Nxp]. Similarly, it is considered
            that for the last (N[0]-Nxp)/2 slices, the molecular orbitals are
            the same than that of N[N[0]-Nxp].
            
    Returns
    -------
        H: tuple of arr
            H[0]: A 1D array whose elements H[0][i] are 2D arrays describing 
            the cross-section Hamiltonian at the position x[i] of the wire.
            H[1]: the 3D Hamiltonian which includes the orbital-coupling terms.
            H[2]: the 3D Hamiltonian which includes the SC-coupling terms
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
    if np.isscalar(m_eff):
        m_eff = m_eff * np.ones((Nx,Ny,Nz))
        
    if np.isscalar(mu):
        mu = mu * np.ones((Nx,Ny,Nz))
    elif not(len(mu[:,0,0])==N[0]):
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
        
    if np.isscalar(B):
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
        
    if np.isscalar(d) and BdG=='yes':
        d = d * np.ones((Nx,Ny,Nz))
        
    ##Built the Hamiltonian:
    #Molecular orbital Hamiltonians
    for i in range(Nx):
        H_aux = scipy.sparse.dok_matrix((int(2 * Ny * Nz),int(2 * Ny * Nz)),dtype=complex)
        if (i>=N_dif) and (i<=Nx-N_dif):
            H_aux = LO_2D_builder_NoSC(N[1::],dis[1::],m_eff[i,:,:],mu[i,:,:],0,np.array([aRx[i,:,:],aRy[i,:,:],aRz[i,:,:]]),space='position',sparse='yes')
            if i==N_dif:
                H_2D=np.array([H_aux.copy()])
            else:
                H_2D=np.append(H_2D,np.array([H_aux.copy()]),axis=0)

    #3D Hamiltonian
    tx=cons.hbar**2/(2*(m_eff[1::,:,:]+m_eff[:-1,:,:])/2*cons.m_e*(dis_x*1e-9)**2)/cons.e*1e3
    e_3D = np.append(2*tx[0,:,:].reshape(1,Ny,Nz),np.append(tx[1::,:,:]+tx[:-1,:,:],2*tx[-1,:,:].reshape(1,Ny,Nz),axis=0),axis=0)
    e_3D,Bx,By,Bz=e_3D.flatten(),Bx.flatten(),By.flatten(),Bz.flatten()
    Bz=np.repeat(Bz,2)
    Bz[1::2] = -Bz[::2]
    tx, aRy_kx, aRz_kx = np.repeat(tx.flatten(),2), np.repeat(((aRy[1::,:,:]+aRy[:-1,:,:])/(4*dis_x)).flatten(),2), ((aRz[1::,:,:]+aRz[:-1,:,:])/(4*dis_x)).flatten()
    aRy_kx[1::2] = -aRy_kx[::2]
    
    H_3D = scipy.sparse.dok_matrix((m,m),dtype=complex)

    H_3D[diagonal(m,k=1,step=2)], H_3D[diagonal(m,k=-1,step=2)] = Bx-1j*By, Bx+1j*By
    H_3D[diagonal(m)] = np.repeat(e_3D,2) + Bz

    H_3D[diagonal(m,k=2*Ny*Nz)] = -tx-1j*aRy_kx
    H_3D[diagonal(m,k=-2*Ny*Nz)] = -tx+1j*aRy_kx
    H_3D[diagonal(m,k=2*Ny*Nz-1,step=2,init=1)] += -aRz_kx
    H_3D[diagonal(m,k=-2*Ny*Nz+1,step=2,init=1)] += -aRz_kx
    H_3D[diagonal(m,k=1+2*Ny*Nz,step=2)] += aRz_kx
    H_3D[diagonal(m,k=-1-2*Ny*Nz,step=2)] += aRz_kx
    
    #SC hybridization
    if BdG=='yes':
        d=d.flatten()
        H_SC = scipy.sparse.dok_matrix((int(2 * Nx * Ny * Nz),int(2 * Nx * Ny * Nz)),dtype=complex)
        H_SC[diagonal(int(2 * Nx * Ny * Nz),k=1,step=2)], H_SC[diagonal(int(2 * Nx * Ny * Nz),k=-1,step=2)] = -np.conj(d), np.conj(d)
        
        return (H_2D,H_3D,H_SC)

    else:
        return (H_2D,H_3D)
    




