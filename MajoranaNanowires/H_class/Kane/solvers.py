
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                  "H_class/Kane/solvers" submodule
                      
This sub-package solves 8-band k.p Hamiltonians for infinite nanowires. Please,
visit http://www.samdaz/MajoranaNanowires.com for more details.

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

from MajoranaNanowires.Functions import diagonal, concatenate


#%%
def Kane_2D_solver(H,N,dis,mu,k_vec,
                   params={},crystal='zincblende',
                   mesh=0,
                   sparse='yes',n_eig=0,near=0,
                   section='rectangular'):
    
    """
    2D 8-band k.p Hamiltonian builder. It obtaines the Hamiltoninan for a 3D
    wire which is infinite in one direction, decribed using 8-band k.p theory.
    
    Parameters
    ----------
        N: int or arr
            Number of sites.
            
        dis: int or arr
            Distance (in nm) between sites.
            
        mu: float or arr
            Chemical potential. If it is an array, each element is the on-site
            chemical potential
            
        k_vec: float or arr
            Momentum along the wire's direction.
            
        params: dic or str
            Kane/Luttinger parameters of the k.p Hamiltonian. 'InAs', 'InSb',
            'GaAs' and 'GaSb' selects the defult parameters for these materials.
            
        crystal: {'zincblende','wurtzite','minimal'}
            Crystal symmetry along the nanowire growth. 'minimal' is a minimal
            model in which the intra-valence band coupling are ignored.
            
        mesh: mesh
            If the discretization is homogeneous, mesh=0. Otherwise, mesh
            provides a mesh with the position of the sites in the mesh.
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
            
        n_eig: int
            Number of sub-band to compute
            
        near: float
            Above which energy must be found the energies of the Hamiltonian.
            
        section: _______
            __________________
           
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
            
        U: arr ((2 x N) x n_eig x n_k)
            Eigenvectors of the system with the same ordering.
    """
    
    if (params=={} or params=='InAs') and crystal=='minimal':
        gamma0, gamma1, gamma2, gamma3 = 1, 0,0,0
        P, m_eff = 919.7, 1.0
        EF, Ecv, Evv, Ep = 0, -417, -390, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2 
        
    elif (params=={} or params=='InSb') and crystal=='minimal':
        gamma0, gamma1, gamma2, gamma3 = 1, 0,0,0
        P, m_eff = 940.2, 1.0
        EF, Ecv, Evv, Ep = 0, -235, -810, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2       


    elif (params=={} or params=='InAs') and (crystal=='zincblende'):
        gamma0, gamma1, gamma2, gamma3 = 1, 20.4, 8.3, 9.1
        P, m_eff = 919.7, 1.0
        EF, Ecv, Evv, Ep = 0, -417, -390, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2
        gamma1, gamma2, gamma3 = gamma1-np.abs(Ep/(3*Ecv)), gamma2-np.abs(Ep/(6*Ecv)), gamma3-np.abs(Ep/(6*Ecv))
        
    elif (params=={} or params=='InSb') and (crystal=='zincblende'):
        gamma0, gamma1, gamma2, gamma3 = 1, 34.8, 15.5, 16.5
        P, m_eff = 940.2, 1.0
        EF, Ecv, Evv, Ep = 0, -235, -810, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2
        gamma1, gamma2, gamma3 = gamma1-np.abs(Ep/(3*Ecv)), gamma2-np.abs(Ep/(6*Ecv)), gamma3-np.abs(Ep/(6*Ecv))    

    elif (params=={} or params=='GaAs') and (crystal=='zincblende'):
        gamma0, gamma1, gamma2, gamma3 = 1, 6.98, 2.06, 2.93
        P, m_eff = 1097.45, 1.0
        EF, Ecv, Evv, Ep = 0, -1519, -341, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2
        Ep=3/(0.063)/(3/np.abs(Ecv)+1/np.abs(Ecv+Evv))
        gamma1, gamma2, gamma3 = gamma1-np.abs(Ep/(3*Ecv)), gamma2-np.abs(Ep/(6*Ecv)), gamma3-np.abs(Ep/(6*Ecv))
        
    elif (params=={} or params=='GaSb') and (crystal=='zincblende'):
        gamma0, gamma1, gamma2, gamma3 = 1, 13.4, 4.7, 6.0
        P, m_eff = 971.3, 1.0
        EF, Ecv, Evv, Ep = 0, -812, -760, (cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2
        gamma1, gamma2, gamma3 = gamma1-np.abs(Ep/(3*Ecv)), gamma2-np.abs(Ep/(6*Ecv)), gamma3-np.abs(Ep/(6*Ecv))
        
    elif (params=={} or params=='InAs') and (crystal=='wurtzite'):
        m_eff = 1.0
        D1,D2,D3,D4=100.3,102.3,104.1,38.8
        A1,A2,A3,A4,A5,A6,A7=-1.5726,-1.6521,-2.6301,0.5126,0.1172,1.3103,-49.04
        B1,B2,B3=-2.3925,2.3155,-1.7231
        e1,e2=-3.2005,0.6363
        P1,P2=838.6,689.87
        alpha1,alpha2,alpha3=-1.89,-28.92,-51.17
        beta1,beta2=-6.95,-21.71
        gamma1,Ec, Ev=53.06,0,-664.9
        
    elif crystal=='minimal' or crystal=='zincblende':
        gamma0, gamma1, gamma2, gamma3 = params['gamma0'], params['gamma1'], params['gamma2'], params['gamma3']
        P, m_eff = params['P'], params['m_eff']
        EF, Ecv, Evv = params['EF'], params['Ecv'], params['Evv']
        
        if crystal=='zincblende':
            Ep=(cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2)**(-1)*P**2
            gamma1, gamma2, gamma3 = gamma1-np.abs(Ep/(3*Ecv)), gamma2-np.abs(Ep/(6*Ecv)), gamma3-np.abs(Ep/(6*Ecv))

    
    ## Make sure that the onsite parameters are arrays:
    Nx, Ny = N[0], N[1]
    if np.ndim(dis)==0:
        dis_x, dis_y = dis, dis
    else:
        dis_x, dis_y = dis[0], dis[1]
    
    if np.isscalar(mesh):
        xi_x, xi_y = np.ones(N), np.ones(N)
    elif len(mesh)==2:
        xi_x, xi_y = dis_x/mesh[0]*np.ones(N), dis_y/mesh[1]*np.ones(N)
    else:
        xi_x, xi_y = dis_x/mesh[0], dis_y/mesh[1]
    
    if np.isscalar(mu):
        mu = mu * np.ones((Nx,Ny))
        
    #Number of bands and sites
    m_b = 8* Nx * Ny
    m_s = Nx * Ny
    if n_eig==0 or sparse=='no':
        n_eig=m_b
    
    if np.isscalar(k_vec):
        k_vec=np.array([k_vec])
    n_k=len(k_vec)
        
    #Obtain the eigenenergies:
    fact=cons.hbar**2/(2*m_eff*cons.m_e)/cons.e*1e3*(1e9)**2
    
    ax=(xi_x[1::,:]+xi_y[:-1,:])/2/(2*dis_x)
    ay=(xi_y[:,1::]+xi_y[:,:-1])/2/(2*dis_y)
    ay=np.insert(ay,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))
    ax,ay=ax.flatten(),ay.flatten()
    mu=mu.flatten()

    #Store matrices:
    E = np.zeros((n_eig,n_k))
    U = np.zeros((m_b,n_eig,n_k),dtype=complex)
        
    
    #Obtain the Hamiltonian:
    if crystal=='zincblende':

        O2p=(concatenate((-ay, ay,-1j*ax,1j*ax)),
            concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
        T=(np.ones(m_s),(diagonal(m_s)))
        
        
        for i in range(n_k):
            print(i)
            ### Upper diagonal:
            ## row 0:
            # (0,5)
            args=T[0]*(-np.sqrt(2/3)*P*k_vec[i])
            index=(T[1][0]+0,T[1][1]+5*m_s)
            
            # (0,6)
            args=np.append(args,T[0]*(-np.sqrt(1/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+6*m_s))

            ## row 1:
            # (1,2)
            args=np.append(args,T[0]*(-np.sqrt(2/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+2*m_s))  

            # (1,7)
            args=np.append(args,T[0]*(np.sqrt(1/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+7*m_s))

            ## row 2:
            # (2.3)
            args=np.append(args,np.conj(O2p[0])*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact)
            index=(np.append(index[0],O2p[1][1]+2*m_s),np.append(index[1],O2p[1][0]+3*m_s))

            # (2.6)
            args=np.append(args,O2p[0]*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact*(-np.sqrt(3/2)))
            index=(np.append(index[0],O2p[1][0]+2*m_s),np.append(index[1],O2p[1][1]+6*m_s))

            # (2.7)
            args=np.append(args,T[0]*2*gamma3*fact*k_vec[i]**2*np.sqrt(2))
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+7*m_s))

            ## row 3:
            # (3,7)
            args=np.append(args,-O2p[0]*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact/np.sqrt(2))
            index=(np.append(index[0],O2p[1][0]+3*m_s),np.append(index[1],O2p[1][1]+7*m_s))

            ## row 4:
            # (4,5)
            args=np.append(args,-np.conj(O2p[0])*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact)
            index=(np.append(index[0],O2p[1][1]+4*m_s),np.append(index[1],O2p[1][0]+5*m_s))  

            # (4,6)
            args=np.append(args,-np.conj(O2p[0])*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact/np.sqrt(2))
            index=(np.append(index[0],O2p[1][1]+4*m_s),np.append(index[1],O2p[1][0]+6*m_s))  

            ## row 5:
            # (5.6)
            args=np.append(args,-T[0]*2*gamma3*fact*k_vec[i]**2*np.sqrt(2))
            index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+6*m_s))
            
            # (5,7)
            args=np.append(args,-np.conj(O2p[0])*2*np.sqrt(3)/3*(2*gamma2+gamma3)*k_vec[i]*fact*np.sqrt(3.0/2))
            index=(np.append(index[0],O2p[1][1]+5*m_s),np.append(index[1],O2p[1][0]+7*m_s))  

            ### Lower diagonal:
            args=np.append(args,np.conj(args))
            index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
            
            ### Diagonal:
            # (0,0)
            args=np.append(args,T[0]*fact*k_vec[i]**2-mu)
            index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+0)) 
            
            # (1,1)
            args=np.append(args,T[0]*fact*k_vec[i]**2-mu)
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+m_s)) 
            
            # (2,2)
            args=np.append(args, -(gamma1+2*gamma3)*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+2*m_s)) 
            
            # (3,3)
            args=np.append(args, -(gamma1-2*gamma3)*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+3*m_s),np.append(index[1],T[1][1]+3*m_s)) 
            
            # (4,4)
            args=np.append(args,-(gamma1-2*gamma3)*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+4*m_s)) 
            
            # (5,5)
            args=np.append(args,-(gamma1+2*gamma3)*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+5*m_s)) 
            
            # (6,6)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+6*m_s),np.append(index[1],T[1][1]+6*m_s)) 
            
            # (7,7)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+7*m_s),np.append(index[1],T[1][1]+7*m_s)) 
            
            ### Built matrix:
            H_add=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
            if sparse=='no':
                H_add=H_add.todense()
            
            ### Solve the Hamiltonian:
            if sparse=='yes':
                E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(H+H_add,k = n_eig,sigma=near, which='LA',tol=1e-5)

            elif sparse=='no':
                E[:,i], U[:,:,i] = np.linalg.eigh(H+H_add)
                
                

        
    elif crystal=='wurtzite':
        A1,A2,A3,A4,A5,A6=A1*fact,A2*fact,A3*fact,A4*fact,A5*fact,A6*fact
        B1,B2,B3=B1*fact,B2*fact,B3*fact
        e1,e2=e1*fact,e2*fact
        
        Kp=(concatenate((-ay, ay,-1j*ax,1j*ax)),
            concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
        T=(np.ones(m_s),(diagonal(m_s)))
        
        
        for i in range(n_k):
            print(i)
            ### Upper diagonal:
            ## row 0:
            # (0,2)
            args=-A6*np.conj(Kp[0])*k_vec[i]
            index=(Kp[1][1]+0,Kp[1][0]+2*m_s)
            
            # (0,6)
            args=np.append(args,-1j*np.conj(Kp[0])*B3*k_vec[i])
            index=(np.append(index[0],Kp[1][1]+0),np.append(index[1],Kp[1][0]+6*m_s))
            
            ## row 1:
            # (1,2)
            args=np.append(args,A6*Kp[0]*k_vec[i])
            index=(np.append(index[0],Kp[1][0]+m_s),np.append(index[1],Kp[1][1]+2*m_s))

            # (1,5)
            args=np.append(args,1j*np.sqrt(2)*alpha1*k_vec[i]*T[0])
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+5*m_s))

            # (1,6)
            args=np.append(args,1j*Kp[0]*B3*k_vec[i])
            index=(np.append(index[0],Kp[1][0]+m_s),np.append(index[1],Kp[1][1]+6*m_s))
            
            # (1,7)
            args=np.append(args,-np.sqrt(2)*beta1*k_vec[i]*T[0])
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+7*m_s))

            
            ## row 2:
            # (2,4)
            args=np.append(args,-1j*np.sqrt(2)*alpha1*k_vec[i]*T[0])
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+4*m_s))
        
            # (2,6)
            args=np.append(args,(P1*k_vec[i]+1j*B1*k_vec[i]**2)*T[0])
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+6*m_s))
            
            
            ## row 3:
            # (3,5)
            args=np.append(args,A6*Kp[0]*k_vec[i])
            index=(np.append(index[0],Kp[1][0]+3*m_s),np.append(index[1],Kp[1][1]+5*m_s))
            
            # (3,7)
            args=np.append(args,1j*B3*Kp[0]*k_vec[i])
            index=(np.append(index[0],Kp[1][0]+3*m_s),np.append(index[1],Kp[1][1]+7*m_s))
            
            
            ## row 4:
            # (4,5)
            args=np.append(args,-A6*np.conj(Kp[0])*k_vec[i])
            index=(np.append(index[0],Kp[1][1]+4*m_s),np.append(index[1],Kp[1][0]+5*m_s))
            
            # (4,6)
            args=np.append(args,-np.sqrt(2)*beta1*k_vec[i]*T[0])
            index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+6*m_s))            
            
            # (4,7)
            args=np.append(args,-1j*B3*k_vec[i]*np.conj(Kp[0]))
            index=(np.append(index[0],Kp[1][1]+4*m_s),np.append(index[1],Kp[1][0]+7*m_s))


            ## row 5:
            # (5,7)
            args=np.append(args,(P1*k_vec[i]+1j*B1*k_vec[i]**2)*T[0])
            index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+7*m_s))


            ### Lower diagonal:
            args=np.append(args,np.conj(args))
            index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
            
            
            ### Diagonal:
            # (0,0)
            args=np.append(args,(A1+A3)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+0)) 
            
            # (1,1)
            args=np.append(args,(A1+A3)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+m_s)) 
            
            # (2,2)
            args=np.append(args,(A1)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+2*m_s)) 
            
            # (3,3)
            args=np.append(args,(A1+A3)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+3*m_s),np.append(index[1],T[1][1]+3*m_s)) 
            
            # (4,4)
            args=np.append(args,(A1+A3)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+4*m_s)) 
            
            # (5,5)
            args=np.append(args,(A1)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+5*m_s)) 
            
            # (6,6)
            args=np.append(args,(e1)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+6*m_s),np.append(index[1],T[1][1]+6*m_s)) 
            
            # (7,7)
            args=np.append(args,(e1)*k_vec[i]**2*T[0]-mu)
            index=(np.append(index[0],T[1][0]+7*m_s),np.append(index[1],T[1][1]+7*m_s)) 
                
                
            ### Built matrix:
            H_add=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
            if sparse=='no':
                H_add=H_add.todense()
            
            ### Solve the Hamiltonian:
            if sparse=='yes':
                E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(H+H_add,k = n_eig,sigma=near, which='LA',tol=1e-5)

            elif sparse=='no':
                E[:,i], U[:,:,i] = np.linalg.eigh(H+H_add)
                
              
    elif crystal=='minimal':

        T=(np.ones(m_s),(diagonal(m_s)))
        
        for i in range(n_k):
            print(i)
            ### Upper diagonal:
            ## row 0:
            # (0,5)
            args=T[0]*(-np.sqrt(2/3)*P*k_vec[i])
            index=(T[1][0]+0,T[1][1]+5*m_s)
            
            # (0,6)
            args=np.append(args,T[0]*(-np.sqrt(1/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+6*m_s))
            
            ## row 1:
            # (1,2)
            args=np.append(args,T[0]*(-np.sqrt(2/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+2*m_s))  

            # (1,7)
            args=np.append(args,T[0]*(np.sqrt(1/3)*P*k_vec[i]))
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+7*m_s))

            ### Lower diagonal:
            args=np.append(args,np.conj(args))
            index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
            
            ### Diagonal:
            # (0,0)
            args=np.append(args,gamma0*T[0]*fact*k_vec[i]**2-mu)
            index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+0)) 
            
            # (1,1)
            args=np.append(args,gamma0*T[0]*fact*k_vec[i]**2-mu)
            index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+m_s)) 
            
            # (2,2)
            args=np.append(args, -gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+2*m_s)) 
            
            # (3,3)
            args=np.append(args, -gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+3*m_s),np.append(index[1],T[1][1]+3*m_s)) 
            
            # (4,4)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+4*m_s)) 
            
            # (5,5)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+5*m_s)) 
            
            # (6,6)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+6*m_s),np.append(index[1],T[1][1]+6*m_s)) 
            
            # (7,7)
            args=np.append(args,-gamma1*T[0]*fact*k_vec[i]**2 - mu)
            index=(np.append(index[0],T[1][0]+7*m_s),np.append(index[1],T[1][1]+7*m_s)) 
            
            ### Built matrix:
            H_add=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
            if sparse=='no':
                H_add=H_add.todense()
            
            ### Solve the Hamiltonian:
            if sparse=='yes':
                E[:,i], U[:,:,i] = scipy.sparse.linalg.eigsh(H+H_add,k = n_eig,sigma=near, which='LA',tol=1e-5)

            elif sparse=='no':
                E[:,i], U[:,:,i] = np.linalg.eigh(H+H_add)
                


    return (E), (U)


