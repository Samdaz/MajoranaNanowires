
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                        "Function" submodule
                      
This sub-package contains some functions required for the "Hamiltonian" 
sub-package. Please, visit http://www.samdaz/MajoranaNanowires.com for more
details.

###############################################################################
           
'''

#%%############################################################################
########################    Required Packages      ############################   
###############################################################################

import numpy as np

import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from scipy import constants

from MajoranaNanowires.third_functions import pfaffian as pf


#%% ############################# Functions
#%%
def FermiDirac(E,kT,mu=0):
    """
    Computes the Fermi-Dirac distribution.
    
    Parameters
    ----------
        E: scalar or arr
            Energies.
        
        kT: scalar
            Temperature.
            
        mu: scalar or arr
            Fermi energy.
            
    Returns
    -------
        result: scalar or arr
            Fermi-Dirac distribution for the given energies.
            
    """
    np.seterr(over='ignore')
    np.seterr(divide='ignore')
    return (1/(1+np.exp((E-mu)/kT)))



#%%
def density_TF(phi,kT=0,E_F=0,material='InAs',band='conduction',Vz=0):
    """
    Computes the charge density of a 3D electron gas in the Thomas-Fermi 
    approximation.
    
    Parameters
    ----------
        phi: scalar or arr
            Electrostatic energy.
        
        kT: scalar
            Temperature.
            
        E_F: scalar or arr
            Fermi energy.
            
        material: str or dic
            Material in which it is evaluated. For a general material,
            'material' is dictionary with arguments m_eff (conduction effective
            mass), m_eff_hh (heavy hole effective mass), m_eff_lh (light hole
            effective mass), and E_gap (semiconductor gap). These parameters
            are already saved in this function for InAs and InSb, which can be
            chosen by choosing material='InAs' or 'InSb', resprectively.
            
        band: str
            Whether to include 'conduction', 'valence' or 'both' bands in the 
            calculations.
            
        Vz: scalar
            Zeeman splitting.
            
    Returns
    -------
        den: scalar or arr
            Charge density in the Thomas-Fermi approximation for the given
            electrostatic energies.
            
    """
    
    np.seterr(invalid='ignore')

    if material=='InAs':
        m_eff=0.023
        m_eff_hh=0.41
        m_eff_lh=0.026
        E_gap=418
    elif material=='InSb':
        m_eff=0.015
        m_eff_hh=0.43
        m_eff_lh=0.015
        E_gap=170
    else:
        if 'E_gap' in material:
            material['m_eff'], material['m_eff_hh'], material['m_eff_lh'], material['E_gap'] = m_eff, m_eff_hh, m_eff_lh, E_gap
        else:
            material['m_eff'] = m_eff

    if band=='conduction':
        if Vz==0:
            den_e=-1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F)*1e-3*constants.e*FermiDirac(-phi-E_F,kT))/constants.hbar)**3*1e-27
            den=np.nan_to_num(den_e,0)
        else:
            den_e=-1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F+Vz)*1e-3*constants.e*FermiDirac(-phi-E_F-Vz,kT))/constants.hbar)**3*1e-27
            den_e=den_e-1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F-Vz)*1e-3*constants.e*FermiDirac(-phi-E_F+Vz,kT))/constants.hbar)**3*1e-27
            den=np.nan_to_num(den_e,0)
        
    elif band=='valence':
        if Vz==0:
            den_hh=1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap,kT))/constants.hbar)**3*1e-27
            den_lh=1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap,kT))/constants.hbar)**3*1e-27
            den=np.nan_to_num(den_hh+den_lh,0)        
        else:
            den_hh=1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F-Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap+Vz,kT))/constants.hbar)**3*1e-27
            den_hh=den_hh+1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F+Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap-Vz,kT))/constants.hbar)**3*1e-27
            den_lh=1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F-Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap+Vz,kT))/constants.hbar)**3*1e-27
            den_lh=den_lh+1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F+Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap-Vz,kT))/constants.hbar)**3*1e-27
            den=np.nan_to_num(den_hh+den_lh,0)
    
    elif band=='both':
        if Vz==0:
            den_e=-1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F)*1e-3*constants.e*FermiDirac(-phi-E_F,kT))/constants.hbar)**3*1e-27
            den_e=np.nan_to_num(den_e,0)
            den_hh=1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap,kT))/constants.hbar)**3*1e-27
            den_lh=1.0/(3*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap,kT))/constants.hbar)**3*1e-27
            den_h=np.nan_to_num(den_hh+den_lh,0)
            den=den_e+den_h
        else:
            den_e=-1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F+Vz)*1e-3*constants.e*FermiDirac(-phi-E_F-Vz,kT))/constants.hbar)**3*1e-27
            den_e=den_e-1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff*constants.m_e*np.abs(phi+E_F-Vz)*1e-3*constants.e*FermiDirac(-phi-E_F+Vz,kT))/constants.hbar)**3*1e-27
            den_e=np.nan_to_num(den_e,0)
            den_hh=1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F-Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap+Vz,kT))/constants.hbar)**3*1e-27
            den_hh=den_hh+1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_hh*constants.m_e*np.abs(-phi-E_gap-E_F+Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap-Vz,kT))/constants.hbar)**3*1e-27
            den_lh=1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F-Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap+Vz,kT))/constants.hbar)**3*1e-27
            den_lh=den_lh+1.0/(6*constants.pi**2)*(np.sqrt(2*m_eff_lh*constants.m_e*np.abs(-phi-E_gap-E_F+Vz)*1e-3*constants.e*FermiDirac(phi+E_F+E_gap-Vz,kT))/constants.hbar)**3*1e-27
            den_h=np.nan_to_num(den_hh+den_lh,0)
            den=den_e+den_h
    
    return (den)


#%% ############################# Array manipulation
#%%
def order_eig(E,U,sparse='yes',BdG='yes'):
    """
    Order the eigenfunctions from smaller to larger. If BdG==yes and
    sparse==yes, it also ensures that there are the same number of positive
    eigenvalues than negative.
    
    Parameters
    ----------
        E: arr
            Eigenvalues.
        
        U: arr
            Eigenvectors.
            
        sparse: {'yes','no'}
            Whether the eigenspectrum has been computed from a sparse matrix.
            
        BdG: {'yes','no'}
            Whether the eigenspectrum must have BdG symmetric or not.
            
            
    Returns
    -------
        E, U: arrs
            Eigenspectrum ordered from smaller to larger eigenvalues.
            
    """
    
    n_eig=len(E)
    
    if BdG=='yes':
        if sparse=='yes':
            idx = np.argsort(E) 
            E = E[idx]    
            U = U[:,idx]
            
            if (np.abs(E[0]+E[n_eig-1])>0.00001)and(np.sign(E[0]+E[n_eig-1])==1):
                E[n_eig-1]=-E[n_eig-2]
            elif (np.abs(E[0]+E[n_eig-1])>0.00001)and(np.sign(E[0]+E[n_eig-1])==-1):
                E[0]=-E[1]
        
    idx = np.argsort(E)     
    E = E[idx]    
    U = U[:,idx]
    
    return (E),(U)



#%%
def length(vec):
    """
    Length of a given vector. If vec is an scalar, its length is 1.
    
    Parameters
    ----------
        vec: scalar or arr
            Input vector            
            
    Returns
    -------
        length: int
            Length of vec. If vec is an scalar, its length is 1.
            
    """
    
    if np.ndim(vec)==0:
        length=1
    else:
        length=len(vec)
    
    return length
    

#%%
def diagonal(N,k=0,init=0,step=1):
    """
    Indices of some diagonal of a marix. It is more efficient than numpy
    counterpart.
    
    Parameters
    ----------
        N: int
            Length of the diagonal (number of elements).
        
        k: int
            Offset of the off-diagonal. k=0 is the main diagonal, k>0 is a
            diagonal in the upper-part of the Hamiltonian, and k<0 in the
            lower one.
            
        init: int
            The starting element of the diagonal.
            
        step: int
            The step between elements in the diagonal.
            
            
    Returns
    -------
        indices: tuple of arr
            Indices of the diagonal. The first element of the tuple are the
            row elements, and the second one are the column ones.
            
    """
    
    assert np.isscalar(k), 'The offset k must be a scalar'
    
    if k==0:
        indices=(np.arange(init,N,step=step),np.arange(init,N,step=step))
    elif k>0:
        indices=(np.arange(init,N-k,step=step),np.arange(init,N-k,step=step)+k)
    elif k<0:
        indices=(np.arange(init,N+k,step=step)-k,np.arange(init,N+k,step=step))
    
    return(indices)
    

#%%
def concatenate(arg):
    """
    Concatenate a list of arrays.
    
    Parameters
    ----------
        arg: tuple or list of arr
            List of arrays to be concatenated.            
            
    Returns
    -------
        con: arr or list
            Array or list of the concatenated list.
            
    """
    
    if isinstance(arg[0],tuple) and len(arg[0])==2:
        index_1, index_2 = np.array([]), np.array([])
        for i in range(len(arg)):
            index_1 = np.append(index_1,arg[i][0])
            index_2 = np.append(index_2,arg[i][1])
        indices=(index_1,index_2)
        
    else:
        indices=np.concatenate(arg)
    
    return(indices) 
    
#%%
def between(arg, interval):
    """
    Computes whether a given number is between a given interval or not.
    
    Parameters
    ----------
        arg: scalar
            Number to be evaluated.
        
        interval: tuple
            Interval in which perform the evaluation.            
            
    Returns
    -------
        result: bool
            If arg is between interval, result=True, and result=False in other
            case.
            
    """
    
    if arg>=interval[0] and arg<=interval[1]:
        result=True
    else:
        result=False
    
    return(result)


#%%
def arg_isclose(vec,val):
    """
    Find the index of a given vector that corresponds to the element closer to
    to an specific value.
    
    Parameters
    ----------
        vec: arr
            Array in which it is desired to find the closer element. 
        
        val: scalar
            Closer value. 
            
    Returns
    -------
        result: int
            Index of the element of vec closer to val.
            
    """
    arg=np.argmin(np.abs(vec-val))
    
    return(arg)

#%% ############################# Constructors or extractors
#%%
def build_mesh(N,L,mesh_type='regular',fact=0.5,asym=1):
    """
    Build a 2D inhomogeneous rectangular mesh.
    
    Parameters
    ----------
        N: arr
            Number of sites in each direction.
        
        L: arr
            Length en each direction.
            
        mesh_type: str
            Whether to build a 'regular' mesh, or an inhomogeneous one with a
            discretization given by a 'geometric' distribution, an 'exponential'
            separation, or a 'random' one.
            
        fact: scalar
            Factor which regulates the separations between sites.
            
        asym: scalar
            The asymetry between the factors applied for the x and y direction.
            
    Returns
    -------
        x, y: mesh
            Mesh in the x and y directions.
            
        dis: mesh
            Mesh with the discretization in each point.
            
    """    

    
    if mesh_type=='regular':
        x, y = np.linspace(-L[1]/2,L[1]/2,N[0]), np.linspace(-L[0]/2,L[0]/2,N[1])
        dis=np.array([np.abs(x[1]-x[0]),np.abs(y[1]-y[0])])
        
        x,y=np.meshgrid(x,y,indexing='ij')
        
        return (x,y,dis)
    
    
    elif mesh_type=='geometric':
        xm,ym=np.zeros(N), np.zeros(N)
        dis_m=np.array([np.zeros(N),np.zeros(N)])
        for i in range(N[0]):
            for j in range(N[1]):
                xm[i,j]=(L[0]/2*fact**np.abs(i-int((N[0]-1)/2))-L[0]/2)*np.sign(i-int((N[0]-1)/2))*(L[0]/(L[0]/2*fact**np.abs(0-int((N[0]-1)/2))-L[0]/2)/2)
                ym[i,j]=(L[1]/2*fact**np.abs(j-int((N[1]-1)/2))-L[1]/2)*np.sign(j-int((N[1]-1)/2))*(L[1]/(L[1]/2*fact**np.abs(0-int((N[1]-1)/2))-L[1]/2)/2)
                    
        for i in range(N[0]):
            for j in range(N[1]):
                if not(j==0 or j==N[1]-1):
                    dis_m[1,i,j]=np.abs(ym[i,j+1]-ym[i,j])/2+np.abs(ym[i,j-1]-ym[i,j])/2
                if not(i==0 or i==N[0]-1):
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])/2+np.abs(xm[i,j]-xm[i+1,j])/2
                if i==0:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i+1,j])
                elif i==N[0]-1:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])
                if j==0:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j+1])
                elif j==N[1]-1:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j-1])

                    
        return (xm,ym,dis_m)
    
    
    elif mesh_type=='exponential':
        np.seterr(all='ignore')
        xm,ym=np.zeros(N), np.zeros(N)
        dis_m=np.array([np.zeros(N),np.zeros(N)])
        for i in range(N[0]):
            for j in range(N[1]):
                xm[i,j]=(1-np.exp(-np.abs(i-int((N[0]-1)/2))*fact))*np.sign(i-int((N[0]-1)/2))*(1-np.exp(-np.abs(N[0]-int((N[0]-1)/2))*fact))**(-1)*L[0]/2
                ym[i,j]=(1-np.exp(-np.abs(j-int((N[1]-1)/2))*fact/asym))*np.sign(j-int((N[1]-1)/2))*(1-np.exp(-np.abs(N[1]-int((N[1]-1)/2))*fact/asym))**(-1)*L[1]/2
                    
        for i in range(N[0]):
            for j in range(N[1]):
                if not(j==0 or j==N[1]-1):
                    dis_m[1,i,j]=np.abs(ym[i,j+1]-ym[i,j])/2+np.abs(ym[i,j-1]-ym[i,j])/2
                if not(i==0 or i==N[0]-1):
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])/2+np.abs(xm[i,j]-xm[i+1,j])/2
                if i==0:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i+1,j])
                elif i==N[0]-1:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])
                if j==0:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j+1])
                elif j==N[1]-1:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j-1])

                    
        return (xm,ym,dis_m)
        
    
    elif mesh_type=='random':
        x,y,dis=build_mesh(N,L,mesh_type='regular')
        
        xm,ym=np.zeros(N), np.zeros(N)
        dis_m=np.array([np.zeros(N),np.zeros(N)])
        for i in range(N[0]):
            for j in range(N[1]):
                xp, yp = x[:,0]+(np.random.rand(N[0])-0.5)*dis[0]*fact, y[0,:]+(np.random.rand(N[0])-0.5)*dis[1]*fact
                xm[i,j],ym[i,j]=xp[i],yp[j]
        for i in range(N[0]):
            for j in range(N[1]):
                if not(j==0 or j==N[1]-1):
                    dis_m[1,i,j]=np.abs(ym[i,j+1]-ym[i,j])/2+np.abs(ym[i,j-1]-ym[i,j])/2
                if not(i==0 or i==N[0]-1):
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])/2+np.abs(xm[i,j]-xm[i+1,j])/2
                if i==0:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i+1,j])
                elif i==N[0]-1:
                    dis_m[0,i,j]=np.abs(xm[i,j]-xm[i-1,j])
                if j==0:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j+1])
                elif j==N[1]-1:
                    dis_m[1,i,j]=np.abs(ym[i,j]-ym[i,j-1])
  
        return (xm,ym,dis_m)


#%%
def get_potential(phi_in,x,y,z,symmetry='none',mesh_type='none'):
    """
    Obtain the potential from a function for a given sites.
    
    Parameters
    ----------
        phi_in: fun
            Fenics function of the electrostatic potential.
        
        x,y,z: arr
            Points in which evaluate the potential.
            
        symmetry: {'none','x','y','z','full-shell'}
            Imposed symmetry of the potential.
        
        mesh_type:
            
    Returns
    -------
        phi_out: arr
            Electrostatic potential in the sites given by x,y,z.
            
    """
    
    phi_out=np.zeros((len(x),len(y),len(z)))
    
    if symmetry=='none':
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
    elif symmetry=='y':
        if mesh_type=='none':
            for i in range(len(x)):
                for j in range(int((len(y)-1)/2)+1):
                    for k in range(len(z)):
                        phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
                        phi_out[i,len(y)-j-1,k]=phi_out[i,j,k]
        elif mesh_type=='yz-mesh':
            for i in range(len(x)):
                for j in range(int((len(y[:,0])-1)/2)+1):
                    for k in range(len(z[0,:])):
                        phi_out[i,j,k]=phi_in(x[i],y[j,k],z[j,k])
                        phi_out[i,len(y[:,0])-j-1,k]=phi_out[i,j,k]
                        
    elif symmetry=='yz':
        for i in range(len(x)):
            for j in range(int((len(y)-1)/2)+1):
                for k in range(int((len(z)-1)/2)+1):
                    phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
                    phi_out[i,len(y)-j-1,k]=phi_out[i,j,k]
                    phi_out[i,j,len(z)-k-1]=phi_out[i,j,k]
                    phi_out[i,len(y)-j-1,len(z)-k-1]=phi_out[i,j,k]
            
    elif symmetry=='xy':
        for i in range(int((len(x)-1)/2)+1):
            for j in range(int((len(y)-1)/2)+1):
                for k in range(len(z)):
                    phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
                    phi_out[i,len(y)-j-1,k]=phi_out[i,j,k]
                    phi_out[len(x)-i-1,j,k]=phi_out[i,j,k]
                    phi_out[len(x)-i-1,len(y)-j-1,k]=phi_out[i,j,k]
    elif symmetry=='x':
        for i in range(int((len(x)-1)/2)+1):
            for j in range(len(y)):
                for k in range(len(z)):
                    phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
                    phi_out[len(x)-i-1,j,k]=phi_out[i,j,k]
                    
    elif symmetry=='full-shell':
        for i in range(len(x)):
            for j in range(int((len(y)-1)/2)+1):
                for k in range(len(z)):
                    if (z[k]>=0) and (y[j]<=z[k]/np.tan(np.pi/3)) and (y[j]>=-z[k]/np.tan(np.pi/3)):
                        phi_out[i,j,k]=phi_in(x[i],y[j],z[k])
                        phi_out[i,len(y)-j-1,k]=phi_out[i,j,k]
                        for l in range(1,4):
                            phi_out[i,int(round((j-25)*np.cos(np.pi/3*l)-(k-25)*np.sin(np.pi/3*l)))+25,int(round((j-25)*np.sin(np.pi/3*l)+(k-25)*np.cos(np.pi/3*l)))+25]=phi_out[i,j,k]
                            phi_out[i,int(round((len(y)-j-1-25)*np.cos(np.pi/3*l)-(k-25)*np.sin(np.pi/3*l)))+25,int(round((len(y)-j-1-25)*np.sin(np.pi/3*l)+(k-25)*np.cos(np.pi/3*l)))+25]=phi_out[i,j,k]
            
            for j in range(int((len(y)-1)/2)+1):
                for k in range(len(z)):
                    if phi_out[i,j,k]==0:
                        phi_out[i,j,k]=phi_out[i,int(j+1),k]
            for j in range(int((len(y)-1)/2)+1):
                for k in range(len(z)):
                    if phi_out[i,j,k]==0:
                        phi_out[i,j,k]=phi_out[i,int(j+2),k]
                    phi_out[i,len(y)-j-1,k]=phi_out[i,j,k]                                    
                
    return (phi_out)

#%%
def get_ElectricField(phi,x,y,z):
    """
    Obtain the electric field of a given electrostatic potential.
    
    Parameters
    ----------
        phi: arr
            Electrostatic potential.
        
        x,y,z: arr
            Points in which it is evaluated the potential.
            
            
    Returns
    -------
        E: arr
            Electric field of phi. Each element E[i] is the electric field in 
            each direction.
            
    """
    
    dis=np.array([np.abs(x[1]-x[0]),np.abs(y[1]-y[0]),np.abs(z[1]-z[0])])
    
    if np.ndim(phi)==3:
        Ex, Ey, Ez = np.gradient(phi,dis[0],dis[1],dis[2])
        return (np.array([Ex,Ey,Ez]))
    
    elif np.ndim(phi)==2:
        Ey, Ez = np.gradient(phi,dis[1],dis[2])
        return (np.array([Ey,Ez]))
    
    elif np.ndim(phi)==1:
        Ex = np.gradient(phi,dis)
        return (Ex)



#%% ############################# Modifiers
#%%
def mask_hexagonal(fun_in,y,z,x=0,change=np.nan,mesh_type='regular'):
    """
    Hexagonal mask. This function change the values for those points of fun_in
    which are outside the hexagonal section.
    
    Parameters
    ----------
        fun_in: arr
            Function to be masked.
        
        y,z: arr
            Points of the section in which it is evaluated the function.
           
        x: arr
            Points of the length in which it is evaluated the function. If x=0,
            then it is only evaluated in 2D.
            
        change: value
            Value to which change those points outside of the hexagonal section.
            
    Returns
    -------
        fun_out: arr
            Masked function.
            
    """
    
    if np.isscalar(x):
        if mesh_type=='regular':
            Ny, Nz = len(y), len(z)
            Ly, Lz = y[Ny-1]*2, z[Nz-1]*2
            a0=Ly/2
            b0=a0*np.sin(np.pi/3)
            
            fun_out=np.zeros((len(y),len(z)))
        
            for j in range(Ny):
                for k in range(Nz):
                    if not(between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0))):
                        fun_out[j,k]=change
                    else:
                        fun_out[j,k]=fun_in[j,k]
        else:
            Ny, Nz = len(y[:,0]), len(z[0,:])
            Ly, Lz = y[Ny-1,0]*2, z[0,Nz-1]*2
            a0=Ly/2
            b0=a0*np.sin(np.pi/3)
            
            fun_out=np.zeros((Ny,Nz))
            for j in range(Ny):
                for k in range(Nz):
                    if not(between(z[j,k], (-b0,b0)) and between(z[j,k],(2*b0/a0*y[j,k]-2*b0,-2*b0/a0*y[j,k]+2*b0)) and between(z[j,k],(-2*b0/a0*y[j,k]-2*b0,2*b0/a0*y[j,k]+2*b0))):
                        fun_out[j,k]=change
                    else:
                        fun_out[j,k]=fun_in[j,k]
        fun_out=np.ma.array(fun_out, mask=np.isnan(fun_out))            
        
    else:
        Ny, Nz = len(y), len(z)
        Ly, Lz = y[Ny-1]*2, z[Nz-1]*2
        a0=Ly/2
        b0=a0*np.sin(np.pi/3)
        
        fun_out=np.zeros((len(x),len(y),len(z)))
        
        for j in range(Ny):
            for k in range(Nz):
                if not(between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0))):
                    fun_out[:,j,k]=np.ones(len(x))*change
                else:
                    fun_out[:,j,k]=fun_in[:,j,k]
        fun_out=np.ma.array(fun_out, mask=np.isnan(fun_out))            
    
    return (fun_out)


#%%
def boundary_wire(N,dis):
    
    if len(N)==2:
        N=np.array([1,N[0],N[1]])
        dis=np.array([0,dis[0],dis[1]])
        
    Nx, Ny, Nz = N[0], N[1], N[2]
    Ly, Lz = dis[1]*Ny, dis[2]*Nz
    y, z = np.linspace(-float(Ly)/2,float(Ly)/2,Ny), np.linspace(-float(Lz)/2,float(Lz)/2,Nz)
    a0=float(Ly)/2
    b0=a0*np.sin(np.pi/3)*(Lz/Ly)
    
    m=0
    sites=np.array([])
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                    if (between(z[k], (b0-dis[2],b0))):
                        sites=np.append(sites,m)
                        sites=np.append(sites,m+1)
                    m=m+2
    return (sites)


#%%
def H_rectangular2hexagonal(H,N,dis,BdG='no',output='H',m=0,sparse='yes'):
    """
    Transform a Hamiltonian of a nanwoire with rectangular cross-section to a
    nanowire with an hexagonal one.
    
    Parameters
    ----------
        H: arr
            Hamiltonian with rectangular section.
            
        N: arr
            Number of sites in each direction.
            
        dis: arr
            Discretization in each direction.
            
        BdG: str
            Whether the Hamiltonian has BdG symmetry.
            
        m: int
            Number of sites of the discretized Hamiltonian with the hexagonal
            section.
            
        output: str
            Whether to return the Hamiltonian (output='H'), the number of sites
            of the discretized Hamiltonian with the hexagonal section 
            (output='m_hex'), or the sites that are inside of the nanowire 
            section (output='sites').

    Returns
    -------
        Depends on the parameter output.
            
    """  
    
    if len(N)==2:
        N=np.array([1,N[0],N[1]])
        dis=np.array([0,dis[0],dis[1]])
        
    Nx, Ny, Nz = N[0], N[1], N[2]
    Ly, Lz = dis[1]*Ny, dis[2]*Nz
    y, z = np.linspace(-float(Ly)/2,float(Ly)/2,Ny), np.linspace(-float(Lz)/2,float(Lz)/2,Nz)
    a0=float(Ly)/2
    b0=a0*np.sin(np.pi/3)*(Lz/Ly)
    
    l=0
    if (output=='H'): 
        if BdG=='no':
            if sparse=='yes':
                H_del=scipy.sparse.dok_matrix((m,2*Nx*Ny*Nz),dtype=complex)
            else: 
                H_del=np.zeros((m,2*Nx*Ny*Nz),dtype=complex)
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                            H_del[l,2*(k+(j+i*Ny)*Nz)]=1
                            H_del[l+1,2*(k+(j+i*Ny)*Nz)+1]=1
                            l=l+2
    
                            
        elif BdG=='yes':
            if sparse=='yes':
                H_del=scipy.sparse.dok_matrix((m,4*Nx*Ny*Nz),dtype=complex)
            else: 
                H_del=np.zeros((m,4*Nx*Ny*Nz),dtype=complex)
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                            H_del[l,2*(k+(j+i*Ny)*Nz)]=1
                            H_del[l+1,2*(k+(j+i*Ny)*Nz)+1]=1
                            H_del[l+int(m/2),2*(k+(j+i*Ny)*Nz)+int(2*Nx*Ny*Nz)]=1
                            H_del[l+1+int(m/2),2*(k+(j+i*Ny)*Nz)+1+int(2*Nx*Ny*Nz)]=1
                            l=l+2

        H=H_del.dot(H.dot(H_del.transpose()))    
        return (H)
    
    
    elif (output=='m_hex'):
        m=0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                        m=m+1
        if BdG=='no':
            m=m*2
    
        elif BdG=='yes':
            m=m*4

        return (m)
    
    elif (output=='sites'):
        m=0
        sites=np.array([])
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                        if (between(z[k], (b0-dis[2],b0))):
                            sites=np.append(sites,m)
                        m=m+2

        return (sites)
    
    
#%%
def U_rectangular2hexagonal(U_in,N,dis,BdG='no',m=0):
    """
    Transform a wavefunction of a nanwoire with rectangular cross-section to a
    nanowire with an hexagonal one, erasing to this end the elements of the
    Hamiltonian outside the hexagonal section of the wire.
    
    Parameters
    ----------
        U_in: arr
            Wavefunction of a nanowire with rectangular section.
            
        N: arr
            Number of sites in each direction.
            
        dis: arr
            Discretization in each direction.
            
        BdG: str
            Whether the Hamiltonian has BdG symmetry.
            
        m: int
            Number of sites of the hexagonal cross-section nanowire. It can be
            computed using the function Function.H_rectangular2hexagonal.

    Returns
    -------
        U: arr
            Wavefunction of a nanowire with hexagonal section.
            
    """  
    
    if len(N)==2:
        N=np.array([1,N[0],N[1]])
        dis=np.array([0,dis[0],dis[1]])
        
    if scipy.sparse.issparse(U_in):
        U_in=U_in.todense()
            
    Nx, Ny, Nz = N[0], N[1], N[2]
    Ly, Lz = dis[1]*Ny, dis[2]*Nz
    y, z = np.linspace(-float(Ly)/2,float(Ly)/2,Ny), np.linspace(-float(Lz)/2,float(Lz)/2,Nz)
    a0=float(Ly)/2
    b0=a0*np.sin(np.pi/3)*(Lz/Ly)
    n_eig=np.shape(U_in)[1]
    
    l=0
    if BdG=='no':
        U=np.zeros((m,n_eig),dtype=complex)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                        U[l,:], U[l+1,:] = U_in[2*(k+(j+i*Ny)*Nz),:], U_in[2*(k+(j+i*Ny)*Nz)+1,:]
                        l=l+2
                        
    elif BdG=='yes':
        U=np.zeros((m,n_eig),dtype=complex)
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                        U[l,:], U[l+1,:] = U_in[2*(k+(j+i*Ny)*Nz),:], U_in[2*(k+(j+i*Ny)*Nz)+1,:]
                        U[l+int(m/2),:], U[l+1+int(m/2),:] = U_in[2*(k+(j+i*Ny)*Nz)+int(2*Nx*Ny*Nz),:], U_in[2*(k+(j+i*Ny)*Nz)+1+int(2*Nx*Ny*Nz),:]
                        l=l+2
    
    U=scipy.sparse.dok_matrix(U)          
    return (U)
    

#%%
def U_hexagonal2rectangular(U_in,N,dis,BdG='no',space='position'):
    """
    Transform a wavefunction of a nanwoire with hexagonal cross-section to a
    nanowire with an rectangular one, filling with zeros the new elements 
    outside the hexagonal section of the wire.
    
    Parameters
    ----------
        U_in: arr
            Wavefunction of a nanowire with hexagonal section.
            
        N: arr
            Number of sites in each direction.
            
        dis: arr
            Discretization in each direction.
            
        BdG: str
            Whether the Hamiltonian has BdG symmetry.
            
        space: str
            Whether the wavefunction is in position space or momentum.

    Returns
    -------
        U: arr
            Wavefunction of a nanowire with rectangular section.
            
    """  
    
    if len(N)==2:
        N=np.array([1,N[0],N[1]])
        dis=np.array([0,dis[0],dis[1]])
    
    if space=='momentum':
        Nx, Ny, Nz = N[0], N[1], N[2]
        m=len(U_in[:,0,0])
        n_eig=len(U_in[0,:,0])
        n_k=len(U_in[0,0,:])
        if BdG=='no':
            U_out = np.empty([2*Nx*Ny*Nz,int(n_eig),n_k],dtype=complex)
        elif BdG=='yes':
            U_out = np.empty([4*Nx*Ny*Nz,int(n_eig),n_k],dtype=complex)
        Ly, Lz = dis[1]*Ny, dis[2]*Nz
        y, z = np.linspace(-float(Ly)/2,float(Ly)/2,Ny), np.linspace(-float(Lz)/2,float(Lz)/2,Nz)
        a0=float(Ly)/2
        b0=a0*np.sin(np.pi/3)*(Lz/Ly)
        l=0
        if BdG=='no':
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                            U_out[2*(k+(j+i*Ny)*Nz),:,:]=U_in[l,:,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:,:]=U_in[l+1,:,:]
                            l=l+2
                        else:
                            U_out[2*(k+(j+i*Ny)*Nz),:,:]=np.zeros((n_eig,n_k))
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:,:]=np.zeros((n_eig,n_k))
    
        elif BdG=='yes':
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0)) ):
                            U_out[2*(k+(j+i*Ny)*Nz),:,:]=U_in[l,:,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:,:]=U_in[l+1,:,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+2*Nx*Ny*Nz,:,:]=U_in[l+int(m/2),:,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1+2*Nx*Ny*Nz,:,:]=U_in[l+1+int(m/2),:,:]
                            l=l+2
                        else:
                            U_out[2*(k+(j+i*Ny)*Nz),:,:]=np.zeros((n_eig,n_k))
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:,:]=np.zeros((n_eig,n_k))
                            U_out[2*(k+(j+i*Ny)*Nz)+2*Nx*Ny*Nz,:,:]=np.zeros((n_eig,n_k))
                            U_out[2*(k+(j+i*Ny)*Nz)+1+2*Nx*Ny*Nz,:,:]=np.zeros((n_eig,n_k))
    
    elif space=='position':
        Nx, Ny, Nz = N[0], N[1], N[2]
        m=len(U_in[:,0])
        n_eig=len(U_in[0,:])
        if BdG=='no':
            U_out = np.empty([2*Nx*Ny*Nz,int(n_eig)],dtype=complex)
        elif BdG=='yes':
            U_out = np.empty([4*Nx*Ny*Nz,int(n_eig)],dtype=complex)
        Ly, Lz = dis[1]*Ny, dis[2]*Nz
        y, z = np.linspace(-float(Ly)/2,float(Ly)/2,Ny), np.linspace(-float(Lz)/2,float(Lz)/2,Nz)
        a0=float(Ly)/2
        b0=a0*np.sin(np.pi/3)*(Lz/Ly)
        l=0
        if BdG=='no':
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0))):
                            U_out[2*(k+(j+i*Ny)*Nz),:]=U_in[l,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:]=U_in[l+1,:]
                            l=l+2
                        else:
                            U_out[2*(k+(j+i*Ny)*Nz),:]=np.zeros((n_eig))
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:]=np.zeros((n_eig))
    
        elif BdG=='yes':
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if (between(z[k], (-b0,b0)) and between(z[k],(2*b0/a0*y[j]-2*b0,-2*b0/a0*y[j]+2*b0)) and between(z[k],(-2*b0/a0*y[j]-2*b0,2*b0/a0*y[j]+2*b0))):
                            U_out[2*(k+(j+i*Ny)*Nz),:]=U_in[l,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:]=U_in[l+1,:]
                            U_out[2*(k+(j+i*Ny)*Nz)+2*Nx*Ny*Nz,:]=U_in[l+int(m/2),:]
                            U_out[2*(k+(j+i*Ny)*Nz)+1+2*Nx*Ny*Nz,:]=U_in[l+1+int(m/2),:]
                            l=l+2
                        else:
                            U_out[2*(k+(j+i*Ny)*Nz),:]=np.zeros((n_eig))
                            U_out[2*(k+(j+i*Ny)*Nz)+1,:]=np.zeros((n_eig))
                            U_out[2*(k+(j+i*Ny)*Nz)+2*Nx*Ny*Nz,:]=np.zeros((n_eig))
                            U_out[2*(k+(j+i*Ny)*Nz)+1+2*Nx*Ny*Nz,:]=np.zeros((n_eig))
                    
    return (U_out)
            



#%% ############################# Spectrum
#%%    
def prob(U,N,BdG='yes'):
    """
    Obtains the probability density of a given wavefunction.
    
    Parameters
    ----------
        U: arr
            Wavefunction in a 1D array.
        
        N: int or arr
            Number of sites. Each element of N[i] is the number of sites along
            the direction i. If N is int, then there is just one dimension.
            
        BdG: {'yes','no'}
            Whether the wvaefunction U is written in the BdG formalism.
            
    Returns
    -------
        P: arr
            Probability density of U with the same dimension than N.
            
    """

    P=np.zeros(N)
    
    if BdG=='no':
        P=(np.abs(U[0::2])**2+np.abs(U[1::2])**2).reshape(N)   

    elif BdG=='yes':
        P=(np.abs(U[0:2*np.prod(N):2])**2+np.abs(U[1:2*np.prod(N):2])**2+np.abs(U[2*np.prod(N)::2])**2+np.abs(U[2*np.prod(N)+1::2])**2).reshape(N)   
            
    return (P)

    
#%%
def Qtot(E,U,kT):
    """
    Computes the total charge in the system.
    
    Parameters
    ----------
        E: scalar or arr
            Energies.
            
        U: arr
            Eigenstates corresponding to each energy.
        
        kT: scalar
            Temperature.


    Returns
    -------
        Qtot: scalar
            Total charge in the system.
            
    """

    den=np.dot(U,np.dot(np.diag(1/(1+np.exp(E/kT))),np.transpose(U)))
    Qtot=np.sum(np.diag(den)[0:int(len(E)/2)])

    return Qtot

#%%
def QM(Uodd,Ueven):
    """
    Computes the Majorana charge (wavefunction overlap).
    
    Parameters
    ----------
        Uodd: arr
            Eigenstate of the odd-parity Majorana state.
            
        Uevev: arr
            Eigenstate of the even-parity Majorana state.


    Returns
    -------
        QM: scalar
            Majorana charge (overlap between U_L and U_R).
            
    """

    QM = np.absolute(np.dot(Uodd+Ueven, -1j*(Uodd-Ueven)))
    return QM

#%%
def Density_Matrix(E,U,kT):
    """
    Computes the density matrix of the system.
    
    Parameters
    ----------
        E: scalar or arr
            Energies.
            
        U: arr
            Eigenstates corresponding to each energy.
        
        kT: scalar
            Temperature.


    Returns
    -------
        den: arr
            Density matrix of the system.
            
    """

    den = np.dot(U, np.dot(np.diag(1 / (1 + np.exp(E / kT))), np.transpose(U)))
    return den

#%%
def Density(E,U,N,kT):
    """
    3D charge densisty.
    
    Parameters
    ----------
        E: arr
            Energies.
            
        U: arr
            Eigenstates.

        N: arr
            Number of sites in each direction.
            
        kT: scalar
            Temperature (in meV).

    Returns
    -------
        den: arr (3D)
            Charge density in each site..
            
    """
    
    np.seterr(over='ignore')
    
    if np.ndim(N)==1:
        Nx=N[0]
        Ny=N[1]
        Nz=N[2]
        n_eig=len(E)
    
        den=np.zeros((Nx,Ny,Nz))
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    for m in range(n_eig):
                        den[i,j,k]=den[i,j,k]+(np.abs(U[2*(k+(i*Ny+j)*Nz),m])**2+np.abs(U[2*(k+(i*Ny+j)*Nz)+1,m])**2)*(1 / (1 + np.exp(E[m] / kT)))                         
        #den = np.dot(U, np.transpose(U))
    
    elif np.ndim(N)==0:
        Nx=N
        n_eig=len(E)
    
        den=np.zeros((Nx))
        for i in range(Nx):
            for m in range(n_eig):
                den[i]=den[i]+(np.abs(U[2*i,m])**2+np.abs(U[2*i+1,m])**2)*(1 / (1 + np.exp(E[m] / kT)))                         
        #den = np.dot(U, np.transpose(U))
    
    return den



#%%
def Density_momentum(E,U,k,N,kT):
    """
    Charge densisty of an infnite system in one direction.
    
    Parameters
    ----------
        E: arr
            Energies.
            
        U: arr
            Eigenstates.
        
        k: arr
            Momentum vector.

        N: arr
            Number of sites in each direction.
            
        kT: scalar
            Temperature (in meV).

    Returns
    -------
        den: arr (2D)
            Charge density in each site.
            
    """    
    
    Nx=N[0]
    Ny=N[1]
    Nz=N[2]
    n_eig=len(E)

    if np.ndim(U)==3:
        den=np.zeros((Nx,Ny,Nz))
        for i_x in range(Nx):
            for i_y in range(Ny):
                for i_z in range(Nz):
                    for i_E in range(n_eig):
                        den[i_x,i_y,i_z]=den[i_x,i_y,i_z]+(np.abs(U[int(2*(i_z+(i_y+i_x*Ny)*Nz)),i_E,0])**2+np.abs(U[int(2*(i_z+(i_y+i_x*Ny)*Nz))+1,i_E,0])**2)*denfromDOS(k,E[i_E,:],kT) 

    elif np.ndim(U)==2:
        Nx=1
        den=np.zeros((Ny,Nz))
        i_x=0
        for i_y in range(Ny):
            for i_z in range(Nz):
                for i_E in range(n_eig):
                    den[i_y,i_z]=den[i_y,i_z]+(np.abs(U[int(2*(i_z+(i_y+i_x*Ny)*Nz)),i_E])**2+np.abs(U[int(2*(i_z+(i_y+i_x*Ny)*Nz))+1,i_E])**2)*denfromDOS(k,E[i_E,:],kT) 
    
    return den
    
#%%
def k_F(mu,aR,Vz,m_eff=0.023):
    """
    Find the Fermi momentum for a 1D nanowire.
    
    Parameters
    ----------
        mu: scalar or arr
            Chemical potential.
            
        aR: scalar or arr
            Spin-orbit coupling.
            
        Vz: scalar or arr
            Zeeman splitting.
            
        m_eff: scalar or str
            Effective mass.

    Returns
    -------
        k_F: scalar or arr
            Fermi momentum.
            
    """    
    
    if m_eff=='InAs':
        m_eff=0.023
    elif m_eff=='InSb':
        m_eff=0.015
    
    m=constants.m_e*m_eff
    hbar=constants.hbar
    
    mu,aR,Vz=mu*1e-3*constants.e,aR*1e-12*constants.e,Vz*1e-3*constants.e
    
    kSO=m*aR/hbar**2
    kZ=np.sqrt(2*m*Vz)/hbar
    kmu_p=2*m*mu/hbar**2
    
    kF=np.zeros(2)
    kF[0]=np.sqrt(2*kSO**2+kmu_p+np.sqrt(4*kSO**4+kZ**4+4*kmu_p*kSO**2))
    kF[1]=np.sqrt(2*kSO**2+kmu_p-np.sqrt(4*kSO**4+kZ**4+4*kmu_p*kSO**2))
    kF=kF*1e-9
    
    return (kF)


    

#%%
def DOS(k,E):
    """
    Density of states of a 1D infinite nanowire.
    
    Parameters
    ----------
        k: arr
            momentum vector.
            
        E: arr
            Energies.
            

    Returns
    -------
        DOS: arr
            Density of states.
            
    """    
    
    DOS=np.abs(np.gradient(E,k))**(-1)/np.pi
    DOS[0]=0
    return(DOS)
    
#%%    
def denfromDOS(k,E,kT):
    """
    1D charge denisty of an infinite nanowire.
    
    Parameters
    ----------
        k: arr
            momentum vector.
            
        E: arr
            Energies.
            

    Returns
    -------
        DOS: arr
            Density of states.
            
    """    
    np.seterr(over='ignore')
    
    dos=DOS(k,E)
    
    den=0
    for i in range(len(E)-1):
        den=den+dos[i]*(E[i+1]-E[i])*(1 / (1 + np.exp(E[i] / kT)))
        
    if not(np.abs(k[0])==np.abs(k[-1])):
        den=den*2
            
    return (den)

#%%
def LDOS(P_n,E_n,E_sample,a_0=0.0):
    """
    Local density of states as a function of the energies E_sample.
    
    Parameters
    ----------
        P_n: arr
            Probability density of the wavefunction at a given point for
            different eigensates.
            
        E_n: arr
            Corresponding energies.
            
        E_sample: arr
            Energies in which the LDOS is evaluated.
            
        a_0: float
            Dirac delta characteristic length. If a_0=0 perfect Dirac delta is
            used, while otherwise it is used an analytical expression for the
            Delta with a characteristic width.
            

    Returns
    -------
        LDOS: arr
            Local density of states for a given energies.
            
    """    

    n_n=len(E_n)
    n_out=len(E_sample)
    
    LDOS=np.zeros(n_out)
    
    if a_0==0.0:
        for i in range(n_out-1):
            for j in range(n_n):
                if (E_sample[i+1]>=E_n[j]) and (E_sample[i]<=E_n[j]):
                    LDOS[i]=LDOS[i]+P_n[j]
        return(LDOS)
        
    else:
        if a_0=='none':
            a_0=np.abs(E_sample[0]-E_sample[1])*4
        
        def Dirac_delta(E,En,a_0):
            return np.exp(-((E-En)/a_0)**2)/(np.sqrt(np.pi)*np.abs(a_0))
        
        for i in range(n_out):
            for j in range(n_n):
                LDOS[i]=LDOS[i]+P_n[j]*Dirac_delta(E_sample[i],E_n[j],a_0)
        return (LDOS)



#%%
def dIdV(LDOS,E_sample,kT):
    """
    Differential conductance for a given energies E_sample.
    
    Parameters
    ----------
        LDOS: arr
            Local density of states computed using Functions.LDOS.
            
        E_sample: arr
            Energies in which the dIdV (and LDOS) is evaluated.
            
        kT: float
            Temperature.

    Returns
    -------
        dIdV: arr
            Differential conductance for a given energies.
            
    """    

    def sech(x):
        return 1.0/np.cosh(x)
    
    n=len(E_sample)
    dIdV=np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            dIdV[i]=dIdV[i]+LDOS[j]*sech((E_sample[i]-E_sample[j])/(2*kT))**2 
    return (dIdV)




#%% ############################# Others
#%%
def Chern_number(H_k,k_vec,N):
    """
    Computes the Chern number of a 1D Hamiltonian in k-space.
    
    Parameters
    ----------
        H_k: arr
            1D Hamiltonian in k-space. Each element H_k[:,:,i] is the 
            Hamiltonian evaluated at k_vec[i].
        
        k_vec: arr
            Momentum vector of the first Brillouin zone in which the
            Hamiltonian is evaluated.
            
        N: int
            Number of sites in which the unit cell of the Hamiltonian is 
            discretized.
            
    Returns
    -------
        Ch: int
            Chern number of the given 1D Hamiltonian.
            
    """
    
    Gamma=np.zeros((4*N,4*N),dtype=complex)
    for i in range(N):
        Gamma[2*i:2*i+2,2*i+2*N:2*i+2*N+2]=np.array([[1,0],[0,1]])
        Gamma[2*i+2*N:2*i+2*N+2,2*i:2*i+2]=np.array([[1,0],[0,1]])    
    
    Ch=np.sign(pf.pfaffian(np.dot(Gamma,H_k[:,:,int((len(k_vec)-1)/2)])))*np.sign(pf.pfaffian(np.dot(Gamma,H_k[:,:,int(len(k_vec)-1)])))
    
    return (Ch)


        
#%%
def rho_acc(x,y,z,den_acc_in,n_lattice,r_lattice,superlattice_type='none'):
    """
    Computes the superficial charge density of a nanowire with hexagonal
    section.
    
    Parameters
    ----------
        x,y,z: arr
            Positions of the mesh of the nanowire section.
        
        den_acc_in: scalar
            Magnitude of the accumulation layer.
            
        n_lattice: int
            Number of superlattice cells.
            
        r_lattice: float
            Partial coverage of the SC.
            
        superlattice_type: str
            Whether the superlattice is on top, at the bottom, or there is no
            superlattice (none).
            
    Returns
    -------
        rho_acc: arr
            Charge density inside the wire due to the charge accumulation layer.
            
    """
    
    Nx, Ny, Nz = len(x), len(y), len(z)
    Lx, Ly, Lz = x[Nx-1], y[Ny-1]*2, z[Nz-1]*2
    a0=Ly/2
    b0=a0*np.sin(np.pi/3)
    dis=np.array([np.abs(x[0]-x[1]),np.abs(y[0]-y[1]),np.abs(z[0]-z[1])])
    L_SC, L_0=Lx/n_lattice*r_lattice, Lx/n_lattice*(1-r_lattice)
    
    den_acc_out=np.zeros((Nx,Ny,Nz))

    if superlattice_type=='top':
        den_acc_out[:,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,-b0)]=np.ones((Nx,arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
        for j in range(Nx):
            for i in range(n_lattice+1):
                if (x[j]>=L_SC/2+i*(L_SC+L_0)) and (x[j]<=L_SC/2+L_0+i*(L_SC+L_0)):
                    den_acc_out[j,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,b0)]=np.ones((arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
    elif superlattice_type=='bottom':
        den_acc_out[:,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,b0)]=np.ones((Nx,arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
        for j in range(Nx):
            for i in range(n_lattice+1):
                if (x[j]>=L_SC/2+i*(L_SC+L_0)) and (x[j]<=L_SC/2+L_0+i*(L_SC+L_0)):
                    den_acc_out[j,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,-b0)]=np.ones((arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
    
    elif superlattice_type=='none':
        den_acc_out[:,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,-b0)+1]=np.ones((Nx,arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
        den_acc_out[:,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,b0)-1]=np.ones((Nx,arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
        
    else:
        for j in range(Nx):
            for i in range(n_lattice+1):
                if (x[j]>=L_SC/2+i*(L_SC+L_0)) and (x[j]<=L_SC/2+L_0+i*(L_SC+L_0)):
                    den_acc_out[j,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,-b0)]=np.ones((arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in
                    den_acc_out[j,arg_isclose(y,-a0/2):arg_isclose(y,a0/2)+1,arg_isclose(z,b0)]=np.ones((arg_isclose(y,a0/2)-arg_isclose(y,-a0/2)+1))*den_acc_in


    for k in range(Nz):
        if (z[k]>=-b0) and (z[k]<=0):
            den_acc_out[:,arg_isclose(2*b0/a0*y-2*b0,z[k]-dis[2])+1,k]=np.ones(Nx)*den_acc_in
            den_acc_out[:,arg_isclose(-2*b0/a0*y-2*b0,z[k]-dis[2])-1,k]=np.ones(Nx)*den_acc_in
        elif (z[k]<=b0) and (z[k]>=0):
            den_acc_out[:,arg_isclose(2*b0/a0*y+2*b0,z[k]+dis[2])-1,k]=np.ones(Nx)*den_acc_in
            den_acc_out[:,arg_isclose(-2*b0/a0*y+2*b0,z[k]+dis[2])+1,k]=np.ones(Nx)*den_acc_in
          
    return (den_acc_out)


