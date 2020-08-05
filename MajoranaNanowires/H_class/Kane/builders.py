
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2020)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                  "H_class/Kane/builders" submodule
                      
This sub-package builds 8-band k.p Hamiltonians for infinite nanowires.

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
def Kane_2D_builder(N,dis,mu,B=0,
                    params={},crystal='zincblende',
                    mesh=0,
                    sparse='yes'):

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
            chemical potential.

        B: float
            Magnetic field along the wire's direction.
            
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
           
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
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
    m_b = 8 * Nx * Ny
    m_s = Nx * Ny
    
    #Obtain the eigenenergies:
    tx=cons.hbar**2/(2*m_eff*cons.m_e*(dis_x*1e-9)**2)/cons.e*1e3*(xi_x[1::,:]+xi_x[:-1,:])/2
    ty=cons.hbar**2/(2*m_eff*cons.m_e*(dis_y*1e-9)**2)/cons.e*1e3*(xi_y[:,1::]+xi_y[:,:-1])/2
    txy=cons.hbar**2/(2*m_eff*cons.m_e*(dis_x*1e-9)*(dis_y*1e-9))/cons.e*1e3*np.append(np.zeros((1,Ny)),xi_x[1::,:]+xi_x[:-1,:],axis=0)/2*np.append(np.zeros((Nx,1)),xi_y[:,1::]+xi_y[:,:-1],axis=1)/2
    txy=txy[1::,1::]
    
    ax=(xi_x[1::,:]+xi_x[:-1,:])/2/(2*dis_x)
    ay=(xi_y[:,1::]+xi_y[:,:-1])/2/(2*dis_y)

    e = np.append(2*tx[0,:].reshape(1,Ny),np.append(tx[1::,:]+tx[:-1,:],2*tx[-1,:].reshape(1,Ny),axis=0),axis=0)
    em = e - np.append(2*ty[:,0].reshape(Nx,1),np.append(ty[:,1::]+ty[:,:-1],2*ty[:,-1].reshape(Nx,1),axis=1),axis=1)
    e += np.append(2*ty[:,0].reshape(Nx,1),np.append(ty[:,1::]+ty[:,:-1],2*ty[:,-1].reshape(Nx,1),axis=1),axis=1)

    ty=np.insert(ty,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))
    ay=np.insert(ay,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))
    txy=np.insert(txy,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))        
    e, em, mu, tx, ty = e.flatten(), em.flatten(), mu.flatten(), tx.flatten(), ty.flatten() 
    ax,ay=ax.flatten(),ay.flatten()
    
    if not(B==0):
        x, y = np.zeros(N), np.zeros(N)
        if np.isscalar(mesh) and mesh==0:
            mesh=np.ones((2,Nx,Ny))*dis[0]
        for i in range(Nx):
            for j in range(Ny):
                x[i,j]=np.sum(mesh[0,0:i+1,j])-(Nx-1)*dis_x/2
                y[i,j]=np.sum(mesh[1,i,0:j+1])-(Ny-1)*dis_y/2
        for i in range(int((Nx-1)/2)):
            x[Nx-i-1,:]=-x[i,:]
        x[int((Nx-1)/2),:]=0
        x=x/np.abs(x[0,0])*(Nx-1)*dis_x/2
        for j in range(int((Ny-1)/2)):
            y[:,Ny-j-1]=-y[:,j]
        y[:,int((Ny-1)/2)]=0
        y=y/np.abs(y[0,0])*(Ny-1)*dis_y/2
                
        fact_B=cons.e/cons.hbar*1e-18
        Mx, My = -fact_B*y/2*B, fact_B*x/2*B
        
        Mx_kx, My_ky = (xi_x[1::,:]*Mx[1::,:]+xi_x[:-1,:]*Mx[:-1,:])/2/(2*dis_x), (xi_y[:,1::]*My[:,1::]+xi_y[:,:-1]*My[:,:-1])/2/(2*dis_y)
        My_ky=np.insert(My_ky,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))
        
        Mm_kx, Mm_ky = (xi_x[1::,:]*(Mx[1::,:]-1j*My[1::,:])+xi_x[:-1,:]*(Mx[:-1,:]-1j*My[:-1,:]))/2/(2*dis_x), -(xi_y[:,1::]*(Mx[:,1::]+1j*My[:,1::])+xi_y[:,:-1]*(Mx[:,:-1]+1j*My[:,:-1]))/2/(2*dis_y)
        Mm_ky=np.insert(Mm_ky,np.arange(Ny-1,(Ny-1)*Nx,(Ny-1)),np.zeros(Nx-1))

        Mx, My = Mx.flatten(), My.flatten()
        Mx_kx, My_ky = Mx_kx.flatten(), My_ky.flatten()
        Mm_kx, Mm_ky = Mm_kx.flatten(), Mm_ky.flatten()
    
    
    ## Built the Hamiltonian:
    if crystal=='zincblende':

        T=(concatenate((e,-tx,-tx,-ty,-ty)),
           concatenate((diagonal(m_s),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny),diagonal(m_s,k=1),diagonal(m_s,k=-1))))
        G1=(concatenate((P/np.sqrt(6)*ay,-P/np.sqrt(6)*ay,-1j*P/np.sqrt(6)*ax,1j*P/np.sqrt(6)*ax)),
            concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
        O1=(concatenate(((-1/np.sqrt(3)*(gamma2+2*gamma3))*em,-tx*(-1/np.sqrt(3)*(gamma2+2*gamma3)),-tx*(-1/np.sqrt(3)*(gamma2+2*gamma3)),
                         (ty*(-1/np.sqrt(3)*(gamma2+2*gamma3))),ty*(-1/np.sqrt(3)*(gamma2+2*gamma3)),-1j*txy[0:-1]/2*(-1/np.sqrt(3)*(gamma2+2*gamma3)),
                         (1j*txy/2*(-1/np.sqrt(3)*(gamma2+2*gamma3))),1j*txy/2*(-1/np.sqrt(3)*(gamma2+2*gamma3)),-1j*txy[0:-1]/2*(-1/np.sqrt(3)*(gamma2+2*gamma3)))),
        concatenate((diagonal(m_s),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny),diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny+1),diagonal(m_s,k=Ny-1,init=1),diagonal(m_s,k=-Ny+1,init=1),diagonal(m_s,k=-Ny-1))))

        if not(B==0):
            B_m=((Mx-1j*My),(diagonal(m_s)))
            B_s=(((Mx**2+My**2)*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3),(diagonal(m_s)))
            B_k=(concatenate((-2*1j*My_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              2*1j*My_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              -2*1j*Mx_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              2*1j*Mx_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3)),concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
            
            B_s_m=(((Mx**2-My**2-2*1j*Mx*My)*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3),(diagonal(m_s)))
            B_k_m=(concatenate((2*Mm_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              -2*Mm_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              -2*1j*Mm_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              2*1j*Mm_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3)),concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))


        ### Upper diagonal:
        ## row 0:
        # (0,2)
        args=G1[0]
        index=(G1[1][0]+0,G1[1][1]+2*m_s)
        
        # (0,4)
        args=np.append(args,np.conj(G1[0])*np.sqrt(3))
        index=(np.append(index[0],G1[1][1]+0),np.append(index[1],G1[1][0]+4*m_s))
        
        # (0,7)
        args=np.append(args,G1[0]*np.sqrt(2))
        index=(np.append(index[0],G1[1][0]+0),np.append(index[1],G1[1][1]+7*m_s))
        
        ## row 1:
        # (1,3)
        args=np.append(args,-G1[0]*np.sqrt(3))
        index=(np.append(index[0],G1[1][0]+m_s), np.append(index[1],G1[1][1]+3*m_s))
        
        # (1,5)
        args=np.append(args,-np.conj(G1[0]))
        index=(np.append(index[0],G1[1][1]+m_s),np.append(index[1],G1[1][0]+5*m_s))          
        
        # (1,6)
        args=np.append(args,np.sqrt(2)*np.conj(G1[0]))
        index=(np.append(index[0],G1[1][1]+m_s), np.append(index[1],G1[1][0]+6*m_s))       
        
        ## row 2:
        # (2,4)
        args=np.append(args,O1[0])
        index=(np.append(index[0],O1[1][0]+2*m_s),np.append(index[1],O1[1][1]+4*m_s))    
        
        # (2,7)
        args=np.append(args,-np.sqrt(2)*T[0]*gamma3)
        index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+7*m_s))    
        
        ## row 3:
        # (3,5)
        args=np.append(args,O1[0])
        index=(np.append(index[0],O1[1][0]+3*m_s),np.append(index[1],O1[1][1]+5*m_s))    
        
        # (3,6)
        args=np.append(args,-np.sqrt(2)*np.conj(O1[0]))
        index=(np.append(index[0],O1[1][1]+3*m_s),np.append(index[1],O1[1][0]+6*m_s))    
        
        ## row 4:
        # (4,7)
        args=np.append(args,np.sqrt(2)*np.conj(O1[0]))
        index=(np.append(index[0],O1[1][1]+4*m_s),np.append(index[1],O1[1][0]+7*m_s))    
        
        ## row 5:
        # (5,6)
        args=np.append(args,np.sqrt(2)*T[0]*gamma3)
        index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+6*m_s))    
        
#        # If there is magentic field:
        if not(B==0):
            ## row 0:
            # (0,2)
            args=np.append(args,P/np.sqrt(6)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+0),np.append(index[1],B_m[1][0]+2*m_s))
            
            # (0,4)
            args=np.append(args,P/np.sqrt(2)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+0),np.append(index[1],B_m[1][1]+4*m_s))
            
            # (0,7)
            args=np.append(args,P/np.sqrt(3)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+0),np.append(index[1],B_m[1][0]+7*m_s))
            
            ## row 1:
            # (1,3)
            args=np.append(args,-P/np.sqrt(2)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+m_s),np.append(index[1],B_m[1][0]+3*m_s))
            
            # (1,5)
            args=np.append(args,-P/np.sqrt(6)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+m_s),np.append(index[1],B_m[1][1]+5*m_s))
            
            # (1,6)
            args=np.append(args,P/np.sqrt(3)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+m_s),np.append(index[1],B_m[1][1]+6*m_s))
            
            ## row 2:
            # (2,7)
            args=np.append(args,-np.sqrt(2)*gamma3*B_s[0])
            index=(np.append(index[0],B_s[1][0]+2*m_s),np.append(index[1],B_s[1][1]+7*m_s)) 
            args=np.append(args,-np.sqrt(2)*gamma3*B_k[0])
            index=(np.append(index[0],B_k[1][0]+2*m_s),np.append(index[1],B_k[1][1]+7*m_s))
            
                # (2,4)
            args=np.append(args,-1/np.sqrt(3)*(gamma2+2*gamma3)*B_s_m[0])
            index=(np.append(index[0],B_s_m[1][0]+2*m_s),np.append(index[1],B_s_m[1][1]+4*m_s)) 
            args=np.append(args,-1/np.sqrt(3)*(gamma2+2*gamma3)*B_k_m[0])
            index=(np.append(index[0],B_k_m[1][0]+2*m_s),np.append(index[1],B_k_m[1][1]+4*m_s)) 
            
            ## row 3:
                # (3,5)
            args=np.append(args,-1/np.sqrt(3)*(gamma2+2*gamma3)*B_s_m[0])
            index=(np.append(index[0],B_s_m[1][0]+3*m_s),np.append(index[1],B_s_m[1][1]+5*m_s)) 
            args=np.append(args,-1/np.sqrt(3)*(gamma2+2*gamma3)*B_k_m[0])
            index=(np.append(index[0],B_k_m[1][0]+3*m_s),np.append(index[1],B_k_m[1][1]+5*m_s)) 
            
                # (3,6)
            args=np.append(args,np.sqrt(2/3)*(gamma2+2*gamma3)*np.conj(B_s_m[0]))
            index=(np.append(index[0],B_s_m[1][1]+3*m_s),np.append(index[1],B_s_m[1][0]+6*m_s)) 
            args=np.append(args,np.sqrt(2/3)*(gamma2+2*gamma3)*np.conj(B_k_m[0]))
            index=(np.append(index[0],B_k_m[1][1]+3*m_s),np.append(index[1],B_k_m[1][0]+6*m_s)) 
            
            ## row 4:
                # (4,7)
            args=np.append(args,-np.sqrt(2/3)*(gamma2+2*gamma3)*np.conj(B_s_m[0]))
            index=(np.append(index[0],B_s_m[1][1]+4*m_s),np.append(index[1],B_s_m[1][0]+7*m_s)) 
            args=np.append(args,-np.sqrt(2/3)*(gamma2+2*gamma3)*np.conj(B_k_m[0]))
            index=(np.append(index[0],B_k_m[1][1]+4*m_s),np.append(index[1],B_k_m[1][0]+7*m_s)) 
            
            ## row 5:
            # (5,6)
            args=np.append(args,np.sqrt(2)*gamma3*B_s[0])
            index=(np.append(index[0],B_s[1][0]+5*m_s),np.append(index[1],B_s[1][1]+6*m_s)) 
            args=np.append(args,np.sqrt(2)*gamma3*B_k[0])
            index=(np.append(index[0],B_k[1][0]+5*m_s),np.append(index[1],B_k[1][1]+6*m_s)) 

        
        ### Lower diagonal:
        args=np.append(args,np.conj(args))
        index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
        
        ### Diagonal:
        # (0,0)
        args=np.append(args,T[0])
        index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+0)) 
        
        # (1,1)
        args=np.append(args,T[0])
        index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+m_s)) 
        
        # (2,2)
        args=np.append(args,(gamma3-gamma1)*T[0])
        index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+2*m_s)) 
        
        # (3,3)
        args=np.append(args,-(gamma3+gamma1)*T[0])
        index=(np.append(index[0],T[1][0]+3*m_s),np.append(index[1],T[1][1]+3*m_s)) 
        
        # (4,4)
        args=np.append(args,-(gamma3+gamma1)*T[0])
        index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+4*m_s)) 
        
        # (5,5)
        args=np.append(args,(gamma3-gamma1)*T[0])
        index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+5*m_s)) 
        
        # (6,6)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+6*m_s),np.append(index[1],T[1][1]+6*m_s)) 
        
        # (7,7)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+7*m_s),np.append(index[1],T[1][1]+7*m_s)) 
        
        if not(B==0):
            # (0,0)
            args=np.append(args,B_s[0])
            index=(np.append(index[0],B_s[1][0]+0),np.append(index[1],B_s[1][1]+0)) 
            args=np.append(args,B_k[0])
            index=(np.append(index[0],B_k[1][0]+0),np.append(index[1],B_k[1][1]+0)) 
            
            # (1,1)
            args=np.append(args,B_s[0])
            index=(np.append(index[0],B_s[1][0]+m_s),np.append(index[1],B_s[1][1]+m_s)) 
            args=np.append(args,B_k[0])
            index=(np.append(index[0],B_k[1][0]+m_s),np.append(index[1],B_k[1][1]+m_s)) 
            
            # (2,2)
            args=np.append(args,(gamma3-gamma1)*B_s[0])
            index=(np.append(index[0],B_s[1][0]+2*m_s),np.append(index[1],B_s[1][1]+2*m_s)) 
            args=np.append(args,(gamma3-gamma1)*B_k[0])
            index=(np.append(index[0],B_k[1][0]+2*m_s),np.append(index[1],B_k[1][1]+2*m_s)) 
            
            # (3,3)
            args=np.append(args,-(gamma3+gamma1)*B_s[0])
            index=(np.append(index[0],B_s[1][0]+3*m_s),np.append(index[1],B_s[1][1]+3*m_s)) 
            args=np.append(args,-(gamma3-gamma1)*B_k[0])
            index=(np.append(index[0],B_k[1][0]+3*m_s),np.append(index[1],B_k[1][1]+3*m_s)) 
            
            # (4,4)
            args=np.append(args,-(gamma3+gamma1)*B_s[0])
            index=(np.append(index[0],B_s[1][0]+4*m_s),np.append(index[1],B_s[1][1]+4*m_s)) 
            args=np.append(args,-(gamma3-gamma1)*B_k[0])
            index=(np.append(index[0],B_k[1][0]+4*m_s),np.append(index[1],B_k[1][1]+4*m_s)) 
            
            # (5,5)
            args=np.append(args,(gamma3-gamma1)*B_s[0])
            index=(np.append(index[0],B_s[1][0]+5*m_s),np.append(index[1],B_s[1][1]+5*m_s)) 
            args=np.append(args,(gamma3-gamma1)*B_k[0])
            index=(np.append(index[0],B_k[1][0]+5*m_s),np.append(index[1],B_k[1][1]+5*m_s)) 
            
            # (6,6)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+6*m_s),np.append(index[1],B_s[1][1]+6*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+6*m_s),np.append(index[1],B_k[1][1]+6*m_s)) 
            
            # (7,7)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+7*m_s),np.append(index[1],B_s[1][1]+7*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+7*m_s),np.append(index[1],B_k[1][1]+7*m_s)) 
        


        ### Built matrix:
        H=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
        if sparse=='no':
            H=H.todense()
        
        ### Add potential and band edges:
        H[diagonal(m_b)]+=-np.tile(mu,8) + concatenate((EF*np.ones(2*m_s),Ecv*np.ones(4*m_s),(Ecv+Evv)*np.ones(2*m_s)))
        

        
    elif crystal=='wurtzite':

        Kc=(concatenate((e,-tx,-tx,-ty,-ty)),
           concatenate((diagonal(m_s),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny),diagonal(m_s,k=1),diagonal(m_s,k=-1))))
        Kp=(concatenate((ay,-ay,-1j*ax,1j*ax)),
            concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
        Kpc=(concatenate((em,-tx,-tx,ty,ty,-1j*txy[0:-1]/2,1j*txy/2,1j*txy/2,-1j*txy[0:-1]/2)),
        concatenate((diagonal(m_s),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny),diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny+1),diagonal(m_s,k=Ny-1,init=1),diagonal(m_s,k=-Ny+1,init=1),diagonal(m_s,k=-Ny-1))))

            
        ### Upper diagonal:
        ## row 0:
        # (0,1)
        args=-A5*np.conj(Kpc[0])
        index=(Kpc[1][1]+0,Kpc[1][0]+m_s)
        
        # (0,2)
        args=np.append(args,1j*(A7-alpha1/np.sqrt(2))*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+0),np.append(index[1],Kp[1][0]+2*m_s))

        # (0,4)
        args=np.append(args,-1j*alpha2*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+0),np.append(index[1],Kp[1][0]+4*m_s))
        
        # (0,6)
        args=np.append(args,-(P2-beta1)/np.sqrt(2)*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+0),np.append(index[1],Kp[1][0]+6*m_s))
        
        
        ## row 1:
        # (1,2)
        args=np.append(args,-1j*(A7+alpha1/np.sqrt(2))*Kp[0])
        index=(np.append(index[0],Kp[1][0]+m_s),np.append(index[1],Kp[1][1]+2*m_s))

        # (1,3)
        args=np.append(args,-1j*alpha2*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+m_s),np.append(index[1],Kp[1][0]+3*m_s))

        # (1,5)
        args=np.append(args,np.sqrt(2)*D3*np.ones(m_s))
        index=(np.append(index[0],diagonal(m_s)[0]+m_s),np.append(index[1],diagonal(m_s)[1]+5*m_s))

        # (1,6)
        args=np.append(args,(P2+beta1)/np.sqrt(2)*Kp[0])
        index=(np.append(index[0],Kp[1][0]+m_s),np.append(index[1],Kp[1][1]+6*m_s))
        
        # (1,7)
        args=np.append(args,1j*np.sqrt(2)*D4*np.ones(m_s))
        index=(np.append(index[0],diagonal(m_s)[0]+m_s),np.append(index[1],diagonal(m_s)[1]+7*m_s))

    
        ## row 2:
        # (2,4)
        args=np.append(args,np.sqrt(2)*D3*np.ones(m_s))
        index=(np.append(index[0],diagonal(m_s)[0]+2*m_s),np.append(index[1],diagonal(m_s)[1]+4*m_s))

        # (2,5)
        args=np.append(args,-1j*alpha3*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+2*m_s),np.append(index[1],Kp[1][0]+5*m_s))
        
        # (2,6)
        args=np.append(args, 1j*B2*Kc[0])
        index=(np.append(index[0],Kc[1][0]+2*m_s),np.append(index[1],Kc[1][1]+6*m_s))

        # (2,7)
        args=np.append(args, beta2*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+2*m_s),np.append(index[1],Kp[1][0]+7*m_s))
        
        
        ## row 3:
        # (3,4)
        args=np.append(args,-A5*Kpc[0])
        index=(np.append(index[0],Kpc[1][0]+3*m_s),np.append(index[1],Kpc[1][1]+4*m_s))

        # (3,5)
        args=np.append(args,-1j*(A7-alpha1/np.sqrt(2))*Kp[0])
        index=(np.append(index[0],Kp[1][0]+3*m_s),np.append(index[1],Kp[1][1]+5*m_s))
        
        # (3,7)
        args=np.append(args,(P2-beta1)/np.sqrt(2)*Kp[0])
        index=(np.append(index[0],Kp[1][0]+3*m_s),np.append(index[1],Kp[1][1]+7*m_s))

    
        ## row 4:
        # (4,5)
        args=np.append(args,1j*(A7+alpha1/np.sqrt(2))*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+4*m_s),np.append(index[1],Kp[1][0]+5*m_s))

        # (4,6)
        args=np.append(args,1j*np.sqrt(2)*D4*np.ones(m_s))
        index=(np.append(index[0],diagonal(m_s)[0]+4*m_s),np.append(index[1],diagonal(m_s)[1]+6*m_s))

        # (4,7)
        args=np.append(args,-(P2+beta1)/np.sqrt(2)*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+4*m_s),np.append(index[1],Kp[1][0]+7*m_s))
    
        
        ## row 5:
        # (5,6)
        args=np.append(args,-beta2*Kp[0])
        index=(np.append(index[0],Kp[1][0]+5*m_s),np.append(index[1],Kp[1][1]+6*m_s))

        # (5,7)
        args=np.append(args, 1j*B2*Kc[0])
        index=(np.append(index[0],Kc[1][0]+5*m_s),np.append(index[1],Kc[1][1]+7*m_s))
        
        
        ## row 6:
        # (6,7)
        args=np.append(args,-1j*gamma1*np.conj(Kp[0]))
        index=(np.append(index[0],Kp[1][1]+6*m_s),np.append(index[1],Kp[1][0]+7*m_s))


        ### Lower diagonal:
        args=np.append(args,np.conj(args))
        index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
        

        ### Diagonal:
        # (0,0)
        args=np.append(args,(A2+A4)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+0),np.append(index[1],Kc[1][1]+0)) 
        
        # (1,1)
        args=np.append(args,(A2+A4)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+m_s),np.append(index[1],Kc[1][1]+m_s)) 
        
        # (2,2)
        args=np.append(args,(A2)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+2*m_s),np.append(index[1],Kc[1][1]+2*m_s)) 
        
        # (3,3)
        args=np.append(args,(A2+A4)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+3*m_s),np.append(index[1],Kc[1][1]+3*m_s)) 
        
        # (4,4)
        args=np.append(args,(A2+A4)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+4*m_s),np.append(index[1],Kc[1][1]+4*m_s)) 
        
        # (5,5)
        args=np.append(args,(A2)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+5*m_s),np.append(index[1],Kc[1][1]+5*m_s)) 
        
        # (6,6)
        args=np.append(args,(e2)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+6*m_s),np.append(index[1],Kc[1][1]+6*m_s)) 
        
        # (7,7)
        args=np.append(args,(e2)*Kc[0])
        index=(np.append(index[0],Kc[1][0]+7*m_s),np.append(index[1],Kc[1][1]+7*m_s)) 


        ### Built matrix:
        H=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
        if sparse=='no':
            H=H.todense()
            
        ### Add potential and band edges:
        H[diagonal(m_b)]+=-np.tile(mu,8) + concatenate(((D1+D2+Ev)*np.ones(m_s),(D1-D2+Ev)*np.ones(m_s),(Ev)*np.ones(m_s),
         (D1+D2+Ev)*np.ones(m_s),(D1-D2+Ev)*np.ones(m_s),(Ev)*np.ones(m_s),
         (Ec)*np.ones(m_s),(Ec)*np.ones(m_s)))

            
    elif crystal=='minimal':

        T=(concatenate((e,-tx,-tx,-ty,-ty)),
           concatenate((diagonal(m_s),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny),diagonal(m_s,k=1),diagonal(m_s,k=-1))))
        G1=(concatenate((P/np.sqrt(6)*ay,-P/np.sqrt(6)*ay,-1j*P/np.sqrt(6)*ax,1j*P/np.sqrt(6)*ax)),
            concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
        
        if not(B==0):
            B_m=((Mx-1j*My),(diagonal(m_s)))
            B_s=(((Mx**2+My**2)*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3),(diagonal(m_s)))
            B_k=(concatenate((-2*1j*My_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              2*1j*My_ky*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              -2*1j*Mx_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3,
                              2*1j*Mx_kx*cons.hbar**2/(2*m_eff*cons.m_e*1e-18)/cons.e*1e3)),concatenate((diagonal(m_s,k=1),diagonal(m_s,k=-1),diagonal(m_s,k=Ny),diagonal(m_s,k=-Ny))))
            
        
        ### Upper diagonal:
        ## row 0:
        # (0,2)
        args=G1[0]
        index=(G1[1][0]+0,G1[1][1]+2*m_s)
        
        # (0,4)
        args=np.append(args,np.conj(G1[0])*np.sqrt(3))
        index=(np.append(index[0],G1[1][1]+0),np.append(index[1],G1[1][0]+4*m_s))
        
        # (0,7)
        args=np.append(args,G1[0]*np.sqrt(2))
        index=(np.append(index[0],G1[1][0]+0),np.append(index[1],G1[1][1]+7*m_s))
        
        ## row 1:
        # (1,3)
        args=np.append(args,-G1[0]*np.sqrt(3))
        index=(np.append(index[0],G1[1][0]+m_s), np.append(index[1],G1[1][1]+3*m_s))
        
        # (1,5)
        args=np.append(args,-np.conj(G1[0]))
        index=(np.append(index[0],G1[1][1]+m_s),np.append(index[1],G1[1][0]+5*m_s))          
        
        # (1,6)
        args=np.append(args,np.sqrt(2)*np.conj(G1[0]))
        index=(np.append(index[0],G1[1][1]+m_s), np.append(index[1],G1[1][0]+6*m_s)) 
        
        
        ## If there is magentic field:
        if not(B==0):
            ## row 0:
            # (0,2)
            args=np.append(args,P/np.sqrt(6)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+0),np.append(index[1],B_m[1][0]+2*m_s))
            
            # (0,4)
            args=np.append(args,P/np.sqrt(2)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+0),np.append(index[1],B_m[1][1]+4*m_s))
            
            # (0,7)
            args=np.append(args,P/np.sqrt(3)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+0),np.append(index[1],B_m[1][0]+7*m_s))
            
            ## row 1:
            # (1,3)
            args=np.append(args,-P/np.sqrt(2)*np.conj(B_m[0]))
            index=(np.append(index[0],B_m[1][1]+m_s),np.append(index[1],B_m[1][0]+3*m_s))
            
            # (1,5)
            args=np.append(args,-P/np.sqrt(6)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+m_s),np.append(index[1],B_m[1][1]+5*m_s))
            
            # (1,6)
            args=np.append(args,P/np.sqrt(3)*B_m[0])
            index=(np.append(index[0],B_m[1][0]+m_s),np.append(index[1],B_m[1][1]+6*m_s))

        
        ### Lower diagonal:
        args=np.append(args,np.conj(args))
        index=(np.append(index[0],index[1]),np.append(index[1],index[0]))
        
        ### Diagonal:
        # (0,0)
        args=np.append(args,gamma0*T[0])
        index=(np.append(index[0],T[1][0]+0),np.append(index[1],T[1][1]+0)) 
        
        # (1,1)
        args=np.append(args,gamma0*T[0])
        index=(np.append(index[0],T[1][0]+m_s),np.append(index[1],T[1][1]+m_s)) 
        
        # (2,2)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+2*m_s),np.append(index[1],T[1][1]+2*m_s)) 
        
        # (3,3)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+3*m_s),np.append(index[1],T[1][1]+3*m_s)) 
        
        # (4,4)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+4*m_s),np.append(index[1],T[1][1]+4*m_s)) 
        
        # (5,5)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+5*m_s),np.append(index[1],T[1][1]+5*m_s)) 
        
        # (6,6)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+6*m_s),np.append(index[1],T[1][1]+6*m_s)) 
        
        # (7,7)
        args=np.append(args,-gamma1*T[0])
        index=(np.append(index[0],T[1][0]+7*m_s),np.append(index[1],T[1][1]+7*m_s)) 
        
        if not(B==0):
            # (0,0)
            args=np.append(args,gamma0*B_s[0])
            index=(np.append(index[0],B_s[1][0]+0),np.append(index[1],B_s[1][1]+0)) 
            args=np.append(args,gamma0*B_k[0])
            index=(np.append(index[0],B_k[1][0]+0),np.append(index[1],B_k[1][1]+0)) 
            
            # (1,1)
            args=np.append(args,gamma0*B_s[0])
            index=(np.append(index[0],B_s[1][0]+m_s),np.append(index[1],B_s[1][1]+m_s)) 
            args=np.append(args,gamma0*B_k[0])
            index=(np.append(index[0],B_k[1][0]+m_s),np.append(index[1],B_k[1][1]+m_s)) 
            
            # (2,2)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+2*m_s),np.append(index[1],B_s[1][1]+2*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+2*m_s),np.append(index[1],B_k[1][1]+2*m_s)) 
            
            # (3,3)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+3*m_s),np.append(index[1],B_s[1][1]+3*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+3*m_s),np.append(index[1],B_k[1][1]+3*m_s)) 
            
            # (4,4)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+4*m_s),np.append(index[1],B_s[1][1]+4*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+4*m_s),np.append(index[1],B_k[1][1]+4*m_s)) 
            
            # (5,5)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+5*m_s),np.append(index[1],B_s[1][1]+5*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+5*m_s),np.append(index[1],B_k[1][1]+5*m_s)) 
            
            # (6,6)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+6*m_s),np.append(index[1],B_s[1][1]+6*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+6*m_s),np.append(index[1],B_k[1][1]+6*m_s)) 
            
            # (7,7)
            args=np.append(args,-gamma1*B_s[0])
            index=(np.append(index[0],B_s[1][0]+7*m_s),np.append(index[1],B_s[1][1]+7*m_s)) 
            args=np.append(args,-gamma1*B_k[0])
            index=(np.append(index[0],B_k[1][0]+7*m_s),np.append(index[1],B_k[1][1]+7*m_s)) 
        

        ### Built matrix:
        H=scipy.sparse.csc_matrix((args,index),shape=(m_b,m_b))
        if sparse=='no':
            H=H.todense()
        
        ### Add potential and band edges:
        H[diagonal(m_b)]+=-np.tile(mu,8) + concatenate((EF*np.ones(2*m_s),Ecv*np.ones(4*m_s),(Ecv+Evv)*np.ones(2*m_s)))
                    

    return (H)



