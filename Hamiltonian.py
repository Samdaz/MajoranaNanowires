
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2018)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                      "Hamiltonian" submodule
                      
This sub-package builds and solves Kitaev, Lutchyn-Oreg and  8-band k.p models 
for nanowires. Please, visit http://www.samdaz/MajoranaNanowires.com for more
details.

###############################################################################
           
'''


#%%############################################################################
########################    Required Packages      ############################   
###############################################################################
import os
import multiprocessing
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np

import MajoranaNanowires.H_class.Lutchyn_Oreg.builders
import MajoranaNanowires.H_class.Lutchyn_Oreg.solvers

import MajoranaNanowires.H_class.Kitaev.builders
import MajoranaNanowires.H_class.Kitaev.solvers

import MajoranaNanowires.H_class.Kane.builders
import MajoranaNanowires.H_class.Kane.solvers


 


#%%############################################################################
########################    Kitaev Nanowires      #############################   
###############################################################################

#%%  
def Kitaev_builder(N,mu,t,Δ,sparse='no'):
    """
    Kitaev Hamiltonian builder. It obtaines the Hamiltoninan for a Kitaev chain.
    
    Parameters
    ----------
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
        
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
            
    
    Notes
    -----
        2-dimensional and 3-dimensional Kitaev Hamiltonians are not still
        avialable.    
    """

    #Obtain the dimension of the system:
    if np.isscalar(N):
        n_dim=1
    else:
        n_dim=len(N)
    
    #Error handler
    assert (n_dim<3),"The number of dimensions must be 1<=N<=3."
        
    if not(sparse=='no' or sparse=='yes'):
        print('AssertionError: Please, for sparse argument, choose between {"yes","no"}. "yes" argument has been chosen as default.')
        sparse='yes'
    
    #Compute the Hamiltonian:
    if n_dim==1:
        H=MajoranaNanowires.H_class.Kitaev.builders.Kitaev_1D_builder(N,mu,t,Δ, sparse=sparse)
    else:
        assert (n_dim<=1),"2D and 3D Kitaev chains are not still avialable."
        
    return (H)
        
                    
#%%                  
def Kitaev_solver(H,n=1,mu=0,n_eig='none'):
    """
    Kitaev Hamiltonian solver. It solves the Hamiltonian of a Kitaev chain.
    
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
            
    
    Notes
    -----
        2-dimensional and 3-dimensional Kitaev Hamiltonians are not still
        avialable.
    """
    
    #Obtain the number of sites:
    N=int(len(H)/2)

    #Obtain the dimension of the system:
    if np.isscalar(N):
        n_dim=1
    else:
        n_dim=len(N)
    
    #Obtain the number of desire eigenvalues:
    if n_eig=='none' or n_eig==0:
        if n_dim==1:
            n_eig=2*N
        else:
            for i_dim in range(n_dim):
                n_eig=2*len(N[i_dim])
                
    #Compute the solution:
    if n_dim==1:
        H=MajoranaNanowires.H_class.Kitaev.solvers.Kitaev_1D_solver(H,n,mu=mu,n_eig=n_eig)
    else:
        assert (n_dim>=1),"2D and 3D Kitaev chains are not still avialable."
                    
        




#%%############################################################################
########################    Lutchyn Nanowires      ############################   
###############################################################################
        

#%%   
def LO_builder(N,dis,m_eff,
                    mu,B,aR,
                    BdG='yes', d=0,
                    space='position',k_vec=np.array([]),
                    sparse='yes',
                    method='BF',Nxp=None):
    
    """
    Lutchy-Oreg Hamiltonian builder. It obtaines the Hamiltoninan for a
    Lutchy-Oreg chain.
    
    Parameters
    ----------
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites. If it is an array, each element is
            the distance between sites in each direction.
            
        m_eff: int or arr
            Effective mass. If it is an array (1D, 2D or 3D), each element 
            is the effective mass on each site of the lattice.
        
        mu: float or arr
            Chemical potential. If it is an array (1D, 2D or 3D), each element 
            is the chemical potential on each site of the lattice.
            
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
            
            
        BdG: {"yes","no"}
            If BdG is "yes", it is built the Hamiltonian in the Bogoliubov-de
            Gennes formalism.            
            
        d: float or arr
            Superconductor paring amplitud.
            -If d is a float, d is the Rashba coupling along the y-direction,
            with the same value in every site.
            -If d is an array (1D, 2D or 3D), each element of the array is
            the superconductor paring amplitud in each site.

            
        space: {"position","momentum"}
            Space in which the Hamiltonian is built. "position" means
            real-space (r-space). In this case the boundary conditions are open.
            On the other hand, "momentum" means reciprocal space (k-space). In
            this case the built Hamiltonian corresponds to the Hamiltonian of
            the unit cell, with periodic boundary conditions along the 
            x-direction. For large matrices, build them in real space, and then
            you can diagonalize it in momentum space with the option 
            "position2momentum" in the Lutchy_solver funtion.
            
        k_vec: arr
            If space=='momentum', k_vec is the (discretized) momentum vector,
            usually in the First Brillouin Zone.
            
            
        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
            
        method: {"BF","MO"}
            Aproach used to build (and solve in future steps) the Hamiltonian.
            The possible methods are:
                1. BF: Brute Force- The entire Hamiltonian is built, so it is
                directly diagonalize in the solver.
                2. MO: Molecular Orbitals (decomposition)- The 3D Hamiltonian
                is projected into the molecular orbital basis spanned by the
                sections along the wire. In this case, the returnted
                Hamiltonian is a tuple where: H[0] is a 1D array whose elements
                H[0][i] are 2D arrays describing the cross-section Hamiltonian 
                at the position x[i] of the wire; H[1] is the 3D Hamiltonian
                which includes the orbital-coupling terms; and, if BdG==yes,
                H[2] is the 3D Hamiltonian which includes the SC-coupling terms.
                
        Nxp: int
            (If method=='MO')
            Number of points to compute the molecular orbitals of the H_2D. For
            the remaining (N[0]-Nxp) slices, it is considered that the 
            molecular orbitals corresponding to the first (N[0]-Nxp)/2 slices
            are the same than for the slice N[Nxp]. Similarly, it is considered
            that for the last (N[0]-Nxp)/2 slices, the molecular orbitals are
            the same than that of N[N[0]-Nxp].
            
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    ##Obtain default parameters if there are none given:
    #Obtain k_vec (if necesssary):
    if space=='momentum':
        assert not(k_vec==np.array([])), 'You have to choose the reciprocal-space vector k_vec in which you want to diagonalize the spectrum.'
        
    #Obtain the effective mass:
    if np.isscalar(m_eff) and m_eff=='InAs':
        m_eff=0.023
    elif np.isscalar(m_eff) and m_eff=='InSb':
        m_eff=0.015
        
    #Obtain the SC pairing amplitud:
    if np.isscalar(d) and d=='Al':
        d=0.2
    elif np.isscalar(d) and d=='NbTiN':
        d=0.5

    ##Compute the Hamiltonian:
    if method=='BF':
        if BdG=='no':
            if np.isscalar(N):
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_1D_builder_NoSC(N,dis,m_eff,mu,B,aR, space=space, k_vec=k_vec, sparse=sparse)
            elif len(N)==2:
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_2D_builder_NoSC(N,dis,m_eff,mu,B,aR, space=space, k_vec=k_vec, sparse=sparse)
            elif len(N)==3:
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_3D_builder_NoSC(N,dis,m_eff,mu,B,aR, space=space, k_vec=k_vec, sparse=sparse)
            
        else:
            if np.isscalar(N):
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_1D_builder(N,dis,m_eff,mu,B,aR,d, space=space, k_vec=k_vec, sparse=sparse)
            elif len(N)==2:
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_2D_builder(N,dis,m_eff,mu,B,aR,d, space=space, k_vec=k_vec, sparse=sparse)
            elif len(N)==3:
                H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_3D_builder(N,dis,m_eff,mu,B,aR,d, space=space, k_vec=k_vec, sparse=sparse)
                
    elif method=='MO':
        if len(N)==3:
            H=MajoranaNanowires.H_class.Lutchyn_Oreg.builders.LO_3D_builder_MO(N,dis,m_eff,mu,B,aR,d=d,Nxp=Nxp,BdG=BdG)

    return (H)


#%%
def LO_solver_multiprocessing(H,N,dis,
                                   args,pipe):
    """
    Allows to solve the Hamiltonian using several CPUs.
    
    Parameters
    ----------
        H: arr
            Discretized Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites. If it is an array, each element is
            the distance between sites in each direction.
            
        arg: dictionary
            Dictionary with the keywords arguments of Lutchyn_solver.
            
        pipe: pipe
            Pipe to the corresponding process.
    """
    
    #Send work to a given process:
    E,U=LO_solver(H,N,dis,1,n_CPU=1,
                       mu=args['mu'],B=args['B'],aR=args['aR'],d=args['d'],
                       SC=args['SC'],BdG=args['BdG'],
                       space=args['space'],k_vec=args['k_vec'], m_eff=args['m_eff'],
                       sparse=args['sparse'],n_eig=args['n_eig'], near=args['near'],
                       section=args['section'],
                       method=args['method'],Nxp=args['Nxp'],n_orb=args['n_orb'])
    #Recover output:
    pipe.send((E,U))
    #Close process:
    pipe.close()
    
    return True



#%%
def LO_solver(H,N,dis,
                   n,n_CPU=1,
                   mu=0,B=0,aR=0,d=0,
                   SC={},BdG='yes',
                   space='position',k_vec=np.array([]), m_eff=0.023,
                   sparse='yes',n_eig=None,near=None,
                   section='rectangular',
                   method='BF',Nxp=None,n_orb=None):
    
    """
    Lutchy-Oreg Hamiltonian solver. It solves the Hamiltoninan (built with 
    Lutchyn_builder) of a Lutchy-Oreg chain.
    
    Parameters
    ----------
        H: arr
            Discretized Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites. If it is an array, each element is
            the distance between sites in each direction.
        
        n: int
            Number of times that the Hamiltonian is diagonalized. In each step,
            it is expected that mu, B or aR is different.
            
        n_CPU: int
            Number of CPUs to be used. Each CPU is used to solve one of the n
            steps. Therefore, n must be a multiple of n_CPU.
            
        
        mu: float or arr
            Chemical potential.
            -If mu is a float or a 1D array of length=N, the same chemical 
            potential is added to the Hamiltonian in every diagonalization step. 
            If it is a float, the constant is added in each site, while if it 
            is an array, each element of the array corresponds to the on-site
            chemical potential.
            -If mu is a 1D array of length=n, it is added in each
            diagonalization step i, the constant value mu[i] in every site.
            -If mu is an array of arrays (n x (N)), in each diagonalization 
            step i, it is added to the Hamiltonian the chemical potential 
            mu[i], whose matrix elements are the chemical potential in each
            site of the lattice.
            
        B: float or arr
            Zeeman splitting.
            -If B is a float, the same constant B is added in the x direction
            in each site and in every diagonalization step.
            -If B is a 1D array of length=3, each element of the array is the
            (constant) Zeeman splitting in each direction, which is added in 
            every diagonalization step.
            -If B is a 1D array of length=n, each element of the array B[i] is 
            added in each diagonalization step i in the x-direction.
            -If B is an 2D array (n x 3), in each diagonalization 
            step i, it is added to the Hamiltonian the Zeeman spliting 
            B[i], whose matrix elements are the (constant) Zeeman splitting in 
            the 3 directions.
            
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
            -If aR is an array of arrays (n x 3 x (N)), in each diagonalization 
            step i, it is added to the Hamiltonian the Rashba coupling 
            aR[i], whose matrix elements are the on-site Rashba coupling in the
            3 directions.
            
        d: float or arr
            Superconducting pairing amplitude.
            -If d is a float or a 1D array of length=N, the same SC pairing 
            amplitude is added to the Hamiltonian in every diagonalization step. 
            If it is a float, the constant is added in each site, while if it 
            is an array, each element of the array corresponds to the on-site
            superconductivity.
            -If d is a 1D array of length=n, it is added in each
            diagonalization step i, the constant value d[i] in every site.
            -If d is an array of arrays (n x (N)), in each diagonalization 
            step i, it is added to the Hamiltonian the SC pairing amplitude 
            d[i], whose matrix elements are the on-site superconductivity in
            each site of the lattice.
            
            
        SC: {}
            If the dictionary is not empty, a Superconductor Hamiltonian is
            added to H before diagonalizing it. The elements of the dictionary...
            
        BdG: {"yes","no"}
            If BdG is "yes", it is solved the Hamiltonian in the Bogoliubov-de
            Gennes formalism.            

            
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
            
        m_eff: int 
            Effective mass. It is only necessary in the 2D Hamiltonian when 
            solving in momentum space.
            
            
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
            
            
        method: {"BF","MO"}
            Aproach used to solve the Hamiltonian. The builder should have the
            same method argument.
            The possible methods are:
                1. BF: Brute Force- The 3D Hamiltonain is directly diagonalize
                in the solver.
                2. MO: Molecular Orbitals (decomposition)- The 3D Hamiltonian
                is projected into the molecular orbital basis spanned by the
                sections along the wire. 
                
        Nxp: int
            (If method=='MO')
            Number of points to compute the molecular orbitals of the H_2D. For
            the remaining (N[0]-Nxp) slices, it is considered that the 
            molecular orbitals corresponding to the first (N[0]-Nxp)/2 slices
            are the same than for the slice N[Nxp]. Similarly, it is considered
            that for the last (N[0]-Nxp)/2 slices, the molecular orbitals are
            the same than that of N[N[0]-Nxp].
            
        n_orb: int
            (If method=='MO')
            Number of moelcular orbitals to include in the diagonalization.
           
            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
         
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
            
    """
    
    
    ##Obtain default parameters if there are none given:
    #Obtain the dimension of the system:
    if np.isscalar(N):
        n_dim=1
    else:
        n_dim=len(N)

    #Obtain the dimension of H:
    if BdG=='yes':
        m=4*np.prod(N)
    else:
        m=2*np.prod(N)
        
    #Obtain the number of eigenvalues:
    if sparse=='yes' or method=='MO':
        assert not(n_eig==None), 'You have to choose the number of bands n_eig you want to find.'
    else:
        n_eig=m

    #Obtain the k_vec (if necessary):
    if space=='momentum':
        assert not(k_vec==np.array([])), 'You have to choose the reciprocal-space vector k_vec in which you want to diagonalize the spectrum.'

    #Obtain the effective mass:
    if np.isscalar(m_eff) and m_eff=='InAs':
        m_eff=0.023
    elif np.isscalar(m_eff) and m_eff=='InSb':
        m_eff=0.015
    
    ##Solve the Hamiltonian:
    #For one value:  
    if n==1 and n_CPU==1:
        if BdG=='yes':
            if n_dim==1:
                E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver(H,N,dis,mu=mu,B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
            elif n_dim==2:
                E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver(H,N,dis,m_eff=m_eff,mu=mu,B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
            elif n_dim==3:
                if method=='BF':
                    E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver(H,N,dis,mu=mu,B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                elif method=='MO':
                    E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu,aR=aR,d=d,sparse=sparse,section=section,BdG=BdG)
    
        else:
            if n_dim==1:
                E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver_NoSC(H,N,dis,mu=mu,B=B,aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
            elif n_dim==2:
                E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver_NoSC(H,N,dis,m_eff=m_eff,mu=mu,B=B,aR=aR,SC=SC,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
            elif n_dim==3:
                if method=='BF':
                    E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_NoSC(H,N,dis,mu=mu,B=B,aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                elif method=='MO':
                    E,U=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu,aR=aR,d=d,sparse=sparse,section=section,BdG=BdG)

    #For several (serial):  
    elif not(n==1) and n_CPU==1:
        
        E=np.empty([n_eig,n])
        U=np.empty([m,n_eig,n],dtype=complex)
        
        if not(np.isscalar(mu)) and len(mu)==n:
            for i in range(n):
                if BdG=='yes':
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver(H,N,dis,mu=mu[i],B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver(H,N,dis,m_eff=m_eff,mu=mu[i],B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver(H,N,dis,mu=mu[i],B=B,aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                        elif method=='MO':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu[i],aR=aR,d=d,sparse=sparse,section=section,BdG=BdG)
                
                else:
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver_NoSC(H,N,dis,mu=mu[i],B=B,aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver_NoSC(H,N,dis,m_eff=m_eff,SC=SC,mu=mu[i],B=B,aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_NoSC(H,N,dis,mu=mu[i],B=B,aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                        elif method=='MO':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu[i],aR=aR,d=d,sparse=sparse,section=section,BdG=BdG)

        elif not(np.isscalar(B)) and len(B)==n:
            for i in range(n):
                if BdG=='yes':
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver(H,N,dis,mu=mu,B=B[i],aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver(H,N,dis,m_eff=m_eff,mu=mu,B=B[i],aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver(H,N,dis,mu=mu,B=B[i],aR=aR,d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)

                else:
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver_NoSC(H,N,dis,mu=mu,B=B[i],aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver_NoSC(H,N,dis,m_eff=m_eff,SC=SC,mu=mu,B=B[i],aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_NoSC(H,N,dis,mu=mu,B=B[i],aR=aR,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                
        elif not(np.isscalar(aR)) and len(aR)==n:
            for i in range(n):
                if BdG=='yes':
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver(H,N,dis,mu=mu,B=B,aR=aR[i],d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver(H,N,dis,m_eff=m_eff,mu=mu,B=B,aR=aR[i],d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver(H,N,dis,mu=mu,B=B,aR=aR[i],d=d,space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                        elif method=='MO':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu,aR=aR[i],d=d,sparse=sparse,section=section,BdG=BdG)

                else:
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver_NoSC(H,N,dis,mu=mu,B=B,aR=aR[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver_NoSC(H,N,dis,m_eff=m_eff,SC=SC,mu=mu,B=B,aR=aR[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_NoSC(H,N,dis,mu=mu,B=B,aR=aR[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                        elif method=='MO':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu,aR=aR[i],d=d,sparse=sparse,section=section,BdG=BdG)
                                
        elif BdG=='yes' and not(np.isscalar(d)) and len(d)==n:
            for i in range(n):
                if BdG=='yes':
                    if n_dim==1:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_1D_solver(H,N,dis,mu=mu,B=B,aR=aR,d=d[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near)
                    elif n_dim==2:
                        E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_2D_solver(H,N,dis,m_eff=m_eff,mu=mu,B=B,aR=aR,d=d[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                    elif n_dim==3:
                        if method=='BF':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver(H,N,dis,mu=mu,B=B,aR=aR,d=d[i],space=space,k_vec=k_vec,sparse=sparse,n_eig=n_eig,near=near,section=section)
                        elif method=='MO':
                            E[:,i],U[:,:,i]=MajoranaNanowires.H_class.Lutchyn_Oreg.solvers.LO_3D_solver_MO(H,N,dis,n_eig,n_orb,Nxp=Nxp,mu=mu,aR=aR,d=d[i],sparse=sparse,section=section,BdG=BdG)
            
    #For several (Multiprocessing):
    elif not(n_CPU==1) and not(n==1):
            
        E=np.empty([n_eig,n])
        U=np.empty([m,n_eig,n],dtype=complex)
        args={'mu':mu,'B':B,'aR':aR,'d':d,'SC':SC,'BdG':BdG,'space':space,'k_vec':k_vec,'m_eff':m_eff,'sparse':sparse,'n_eig':n_eig,'near':near,'section':section,'method':method,'Nxp':Nxp,'n_orb':n_orb}
        
        for i_tot in range(int(n/n_CPU)):
            jobs=[]
            pipes=[]
            for i_CPU in range(n_CPU):
                pipe_start, pipe_end = multiprocessing.Pipe()
                pipes.append(pipe_start)
                if not(np.isscalar(mu)) and len(mu)==n:
                    args['mu']=mu[n_CPU*i_tot+i_CPU]
                    thread=multiprocessing.Process(target=LO_solver_multiprocessing,args=(H,N,dis,args,pipe_end,))
                if not(np.isscalar(B)) and len(B)==n:
                    args['B']=B[n_CPU*i_tot+i_CPU]
                    thread=multiprocessing.Process(target=LO_solver_multiprocessing,args=(H,N,dis,args,pipe_end,))
                if not(np.isscalar(aR)) and len(aR)==n:
                    args['aR']=aR[n_CPU*i_tot+i_CPU]
                    thread=multiprocessing.Process(target=LO_solver_multiprocessing,args=(H,N,dis,args,pipe_end,))
                if BdG=='yes' and not(np.isscalar(d)) and len(d)==n:
                    args['d']=d[n_CPU*i_tot+i_CPU]
                    thread=multiprocessing.Process(target=LO_solver_multiprocessing,args=(H,N,dis,args,pipe_end,))
                jobs.append(thread)
                thread.start()
            for i_CPU in range(n_CPU):
                (E[:,n_CPU*i_tot+i_CPU],U[:,:,n_CPU*i_tot+i_CPU])=pipes[i_CPU].recv()
            for i_CPU in jobs:
                i_CPU.join()
                
    return (E),(U)








#%%############################################################################
###########################    Kane Nanowires      ############################   
###############################################################################
        

#%%   
def Kane_builder(N,dis,
                 mu,
                 mesh=0,sparse='yes',
                 params={},crystal='zincblende'):
    
    """
    8-band Kane Hamiltonian builder. It obtaines the Hamiltoninan for a
    8-band Kane Hamiltonain.
    
    Parameters
    ----------
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites. If it is an array, each element is
            the distance between sites in each direction.
        
        mu: float or arr
            Chemical potential. If it is an array (1D, 2D or 3D), each element 
            is the chemical potential on each site of the lattice.
            
        mesh: arr
            Discretization mesh.
            

        sparse: {"yes","no"}
            Sparsety of the built Hamiltonian. "yes" builds a dok_sparse matrix, 
            while "no" builds a dense matrix.
        
        params: Dictionary or str
            Parameters to use for the Kane model. There are some default ones,
            for InAs, InSb, GaAs, and GaSb.
            
        crystal: {'minimal','zincblende','wurtzite'}
            Crystalography of the crystal.
        
            
    Returns
    -------
        H: arr
            Hamiltonian matrix.
    """
    
    ##Compute the Hamiltonian:
    if np.isscalar(N):
        H=MajoranaNanowires.H_class.Kane.builders.Kane_1D_builder(N,dis,mu,mesh=mesh,sparse=sparse,params=params,crystal=crystal)
    elif len(N)==2:
        H=MajoranaNanowires.H_class.Kane.builders.Kane_2D_builder(N,dis,mu,mesh=mesh,sparse=sparse,params=params,crystal=crystal)
    elif len(N)==3:
        H=MajoranaNanowires.H_class.Kane.builders.Kane_3D_builder(N,dis,mu,mesh=mesh,sparse=sparse,params=params,crystal=crystal)
                
    return (H)



#%%
def Kane_solver(H,N,dis,
                mu,k_vec,
                mesh=0,sparse='yes',n_eig=0, near=0,
                params={},crystal='zincblende',section='rectangular'):
    
    
    """
    8-band Kane Hamiltonian solver. It obtaines the eigenspectrum of a 8-band 
    Hamiltoninan (built with Kane_builder).
    
    Parameters
    ----------
        H: arr
            Discretized Lutchyn-Oreg Hamiltonian built with Lutchyn_builder.
            
        N: int or arr
            Number of sites. If it is an array, each element is the number of
            sites in each direction.
            
        dis: int or arr
            Distance (in nm) between sites. If it is an array, each element is
            the distance between sites in each direction.
        
        mu: float or arr
            Chemical potential. If it is a float, the constant is added in each
            site, while if it is an array, each element of the array 
            corresponds to the on-site chemical potential.
            
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
            
            
        params: Dictionary or str
            Parameters to use for the Kane model. There are some default ones,
            for InAs, InSb, GaAs, and GaSb.
            
        crystal: {'minimal','zincblende','wurtzite'}
            Crystalography of the crystal.
        
                
        section: {"rectangular","hexagonal"}
            Whether the system have a rectangular or hexagonal cross-section
            in the plane zy.

            
    Returns
    -------
        E: arr (n_eig x n)
            Eigevalues (energies), ordered from smaller to larger.
         
        U: arr ((2 x N) x n_eig x n)
            Eigenvectors of the system with the same ordering.
            
    """
    
    
    ##Obtain default parameters if there are none given:
    #Obtain the dimension of the system:
    if np.isscalar(N):
        n_dim=1
    else:
        n_dim=len(N)
        
    #Obtain the number of eigenvalues:
    if sparse=='yes':
        assert not(n_eig==None), 'You have to choose the number of bands n_eig you want to find.'
    else:
        n_eig=8*np.prod(N)
    
    #Obtain the k_vec (if necessary):
    assert not(k_vec==np.array([])), 'You have to choose the reciprocal-space vector k_vec in which you want to diagonalize the spectrum.'


    ##Solve the Hamiltonian:
    if n_dim==1:
        E,U=MajoranaNanowires.H_class.Kane.solvers.Kane_1D_solver(H,N,dis,mu,k_vec,mesh=mesh,sparse=sparse,n_eig=n_eig,near=near,params=params,crystal=crystal,section=section)
    elif n_dim==2:
        E,U=MajoranaNanowires.H_class.Kane.solvers.Kane_2D_solver(H,N,dis,mu,k_vec,mesh=mesh,sparse=sparse,n_eig=n_eig,near=near,params=params,crystal=crystal,section=section)
    elif n_dim==3:
        E,U=MajoranaNanowires.H_class.Kane.solvers.Kane_3D_solver(H,N,dis,mu,k_vec,mesh=mesh,sparse=sparse,n_eig=n_eig,near=near,params=params,crystal=crystal,section=section)

                
    return (E),(U)



