
'''
###############################################################################

                  "MajoranaNanowire" Python3 Module
                             v 1.0 (2020)
                Created by Samuel D. Escribano (2018)

###############################################################################
                
                        "Environment" submodule
                      
This sub-package builds and solves the electrostatic environment for different 
heterostructures.

###############################################################################
           
'''


#%%############################################################################
########################    Required Packages      ############################   
###############################################################################
import numpy as np

from scipy import constants as cons
from scipy import interpolate

import dolfin as dl


#%%############################################################################
######################    Environment builders      ###########################
###############################################################################

#%%
def Shell(L_wire,W_wire,W_SiO,W_SC,a_fen,W_0=0,section='rectangular',bc='open',faces=np.array(['1']),mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire covered with a metallic 
    shell. Experiment: Majorana's experiments.
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Width of the metallic shell.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        faces: arr
            Facets that the metallic shell covers the wire. Each facet is
            labeled with a number from 1 to 6 (the upper one is 1, and the
            rest are numbered clockwise). Each element of the array denotes
            with a string (e.g. np.array(['1','2'])) if such facet is covered.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.

            
    """
    
    #Obtain some geometrical parameters:
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)
    R_wire=W_wire/2
    
    if np.ndim(W_0)==0:
        W_0=np.ones(3)*W_wire        
    W_0x, W_0y, W_0z = W_0[0], W_0[1], W_0[2]
        
    ##Build the mesh if none is provided:    
    if (mesh==0)and (V==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0y,-R_wire-W_SiO),dl.Point(L_wire,R_wire+W_0y,R_wire+W_0z), int(L_wire/a_fen+1),int((W_wire+2*W_0y)/a_fen),int((W_wire+W_SiO+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-b0-W_0y,-b0-W_SiO),dl.Point(L_wire,b0+W_0y,b0+W_0z), int(L_wire/a_fen+1),int((2*b0+2*W_0y)/a_fen+1),int((2*b0+W_SiO+W_0z)/a_fen+1))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
            
        #With open boundary conditions:
        else:        
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-R_wire-W_0y,-R_wire-W_SiO),dl.Point(L_wire+W_0x,R_wire+W_0y,R_wire+W_0z), int((2*W_0x+L_wire)/a_fen),int((W_wire+2*W_0y)/a_fen),int((W_wire+W_SiO+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-b0-W_0y,-b0-W_SiO),dl.Point(L_wire+W_0x,b0+W_0y,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*b0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z)/a_fen))
            
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            

    ##Create subdomains:
    #For a rectangular section:
    if section=='rectangular':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-R_wire, -R_wire)))):
                    return True
                else:
                    return False
        
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
        
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-R_wire + 1e-3)
                
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='2').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire,R_wire+W_SC)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='4').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC,-R_wire)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='1').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='3').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (-R_wire-W_SC,-R_wire)))):
                    return True
                elif (faces=='lead_1').any() and ((dl.between(x[0], (-W_SC, 0+dl.DOLFIN_EPS)) and dl.between(x[1], (-R_wire-W_SC,R_wire+W_SC)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='lead_2').any() and ((dl.between(x[0], (L_wire, L_wire+W_SC+dl.DOLFIN_EPS)) and dl.between(x[1], (-R_wire-W_SC,R_wire+W_SC)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                else:
                    return False
                    
    #For a hexagonal section: 
    elif section=='hexagonal':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-b0, -b0)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-b0 + 1e-3)
            
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='1').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0-a_fen,b0+W_SC)))):
                    return True
                elif (faces=='2').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                    return True
                elif (faces=='6').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                    return True
                elif (faces=='3').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_SC,-2*b0/a0*x[1]+2*b0)))):
                    return True
                elif (faces=='5').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                    return True
                elif (faces=='4').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen)))):
                    return True
                elif (faces=='lead_1').any() and ((dl.between(x[0], (-W_SC, 0+dl.DOLFIN_EPS)) and dl.between(x[1], (-R_wire-W_SC,R_wire+W_SC)) and dl.between(x[2], (-b0,b0+W_SC)))):
                    return True
                elif (faces=='lead_2').any() and ((dl.between(x[0], (L_wire, L_wire+W_SC+dl.DOLFIN_EPS)) and dl.between(x[1], (-R_wire-W_SC,R_wire+W_SC)) and dl.between(x[2], (-b0,b0+W_SC)))):
                    return True
                else:
                    return False
                
    #Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    #Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    
    return (mesh),(V),(domains),(boundaries)


#%%    
def Magnetic_layer(L_wire,W_wire,W_sub,W_SC,W_mag,W_ox,a_fen,W_sep,section='rectangular',bc='open',contact='yes',bc_EuS='no',faces='1',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowie partially covered by a 
    metallic shell and a magnetic (insulating) one. Experiment: Y. Liu et al.
    Nano Lett. 20, 456-462 (2020).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_sub: float
            Width of the substrate.
            
        W_SC: float
            Width of the metallic layer.
            
        W_mag: float
            Width of the magnetic layer.
            
        W_ox: float
            Width of the oxidized layer.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_sep: float
            Separation between each side-gate and the corner of the wire.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        contact: {"yes","no"}
            Whether the metallic and magnetic layers overlap.
            
        bc_EuS: {"yes","no"}
            Whether to put a boundary condition in the part of the SC in
            contact with the magnetic layer different to the other facet.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
    """
    
    #Obtain some geometrical parameters:
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)        
    R_wire=W_wire/2
    
    if bc=='open':
        space=L_wire*0.1
    elif bc=='periodic':
        space=0
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):            
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_sep,-R_wire-W_sub),dl.Point(L_wire,R_wire+W_sep,R_wire*1.2+W_SC+W_mag+W_ox), int(L_wire/a_fen+1),int((W_wire+2*W_sep)/a_fen),int((W_wire*1.1+W_SC+W_mag+W_ox)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-a0-W_sep,-b0-W_sub),dl.Point(L_wire,a0+W_sep,b0*1.2+W_SC+W_mag+W_ox), int(L_wire/a_fen+1),int((2*a0+2*W_sep)/a_fen),int((2*b0*1.1+W_SC+W_mag+W_ox)/a_fen))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With open boundary conditions:
        else:        
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_sep,-R_wire-W_sub),dl.Point(L_wire,R_wire+W_sep,R_wire*1.2+W_SC+W_mag+W_ox), int(L_wire/a_fen+1),int((W_wire+2*W_sep)/a_fen),int((W_wire*1.1+W_SC+W_mag+W_ox)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-a0-W_sep,-b0-W_sub),dl.Point(L_wire,a0+W_sep,b0*1.2+W_SC+W_mag+W_ox), int(L_wire/a_fen+1),int((2*a0+2*W_sep)/a_fen),int((2*b0*1.1+W_SC+W_mag+W_ox)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            
    ##Create the domains:
    if section=='rectangular':
        class Substrate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-R_wire, -R_wire)))):
                    return True
                else:
                    return False
        
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class Back_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-R_wire + 1e-3)
            
        class Left_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0+space, L_wire-space)) and (x[1]<=-R_wire-W_sep+1e-3) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class Right_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0+space, L_wire-space)) and (x[1]>=R_wire+W_sep-1e-3) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC,-R_wire)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire,R_wire+W_SC)))):
                        return True
                    else:
                        return False
                    
                elif contact=='yes':
                    if bc_EuS=='no':
                        if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC,-R_wire)) and dl.between(x[2], (-R_wire,R_wire+W_SC+W_mag)))):
                            return True
                        elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire+W_mag,R_wire+W_SC+W_mag)))):
                            return True
                        else:
                            return False
                    elif bc_EuS=='yes':
                        if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC,-R_wire)) and dl.between(x[2], (-R_wire,R_wire+W_SC+W_mag)))):
                            return True
                        else:
                            return False

        if bc_EuS=='yes' and contact=='yes':
            class SC_layers_EuS(dl.SubDomain):
                def inside(self, x, on_boundary):
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire+W_mag,R_wire+W_SC+W_mag)))):
                        return True
                    else:
                        return False          
                    
        class mag_layer(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire,R_wire+W_mag)) and dl.between(x[2], (-R_wire,R_wire)))):
                        return True
                    else:
                        return False
                    
                elif contact=='yes':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire,R_wire+W_mag)) and dl.between(x[2], (-R_wire,R_wire+W_mag)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire,R_wire+W_mag)))):
                        return True
                    else:
                        return False
                    
        class oxide_layer(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire+W_mag,R_wire+W_mag+W_ox)) and dl.between(x[2], (-R_wire,R_wire)))):
                        return True
                    else:
                        return False
                    
                elif contact=='yes':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire+W_mag,R_wire+W_mag+W_ox)) and dl.between(x[2], (-R_wire,R_wire+W_mag+W_SC+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC-W_ox,-R_wire-W_SC)) and dl.between(x[2], (-R_wire,R_wire+W_mag+W_SC+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire+W_mag+W_SC,R_wire+W_mag+W_SC+W_ox)))):
                        return True
                    else:
                        return False
                    
    elif section=='hexagonal':
        class Substrate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-b0, -b0)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Back_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-b0 + 1e-3)
            
        class Left_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0+space, L_wire-space)) and (x[1]<=-a0-W_sep+1e-3) and dl.between(x[2], (-b0, b0))):
                    return True
                else:
                    return False
                
        class Right_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0+space, L_wire-space)) and (x[1]>=a0+W_sep-1e-3) and dl.between(x[2], (-b0, b0))):
                    return True
                else:
                    return False
            
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0,b0+W_SC)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    else:
                        return False
                
                elif contact=='yes':
                    if bc_EuS=='no':
                        if faces=='2':
                            if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0+W_mag,b0+W_mag+W_SC)))):
                                return True
                            elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,b0+W_mag)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                                return True
                            elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                                return True
                            else:
                                return False

                        else:
                            if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0+W_mag,b0+W_mag+W_SC)))):
                                return True
                            elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,b0+W_mag)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                                return True
                            else:
                                return False
                    if bc_EuS=='yes':
                        if ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,b0+W_mag)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                            return True
                        else:
                            return False
                        
        if bc_EuS=='yes' and contact=='yes':
            class SC_layers_EuS(dl.SubDomain):
                def inside(self, x, on_boundary):
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0+W_mag,b0+W_mag+W_SC)))):
                        return True
                    else:
                        return False
                        
                        
        class mag_layer(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,b0)) and dl.between(x[2], (0,-2*b0/a0*x[1]+2*b0+W_mag)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_mag,0)))):
                        return True
                    else:
                        return False
                
                elif contact=='yes':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0,b0+W_mag)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+W_mag)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_mag)))):
                        return True
                    else:
                        return False
                
                    
        class oxide_layer(dl.SubDomain):
            def inside(self, x, on_boundary):
                if contact=='no':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*(x[1]-W_mag)+2*b0-a_fen*b0/a0*2,b0)) and dl.between(x[2], (0,-2*b0/a0*(x[1]-W_mag)+2*b0+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,2*b0/a0*(x[1]-W_mag)-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (2*b0/a0*(x[1]-W_mag)-2*b0-W_ox,0)))):
                        return True
                    else:
                        return False
                    
                elif contact=='yes':
                    if ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0+W_SC+W_mag,b0+W_SC+W_mag+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*(x[1]-W_mag)+2*b0-a_fen*b0/a0*2,b0+W_SC+W_mag+W_ox)) and dl.between(x[2], (2*b0/a0*(x[1]-W_mag)-2*b0,-2*b0/a0*(x[1]-W_mag)+2*b0+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*(x[1]+W_mag)+2*b0-a_fen*b0/a0*2,b0+W_SC+W_mag+W_ox)) and dl.between(x[2], (-2*b0/a0*(x[1]+W_mag)-2*b0,2*b0/a0*(x[1]+W_mag)+2*b0+W_ox)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,2*b0/a0*(x[1])-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (2*b0/a0*(x[1])-2*b0-W_ox,-2*b0/a0*(x[1]-W_mag)+2*b0)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-2*b0/a0*(x[1])-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*(x[1])-2*b0-W_ox,2*b0/a0*(x[1]+W_mag)+2*b0)))):
                        return True
                    elif ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-b0+W_mag)) and (dl.between(x[1], (-W_sep-R_wire,-a0/2-W_ox)) or dl.between(x[1], (a0/2+W_ox,W_sep+R_wire))) )):
                        return True
                    else:
                        return False
                    
    if (bc=='open'):   
        if section=='rectangular':
            class Leads(dl.SubDomain):
                def inside(self, x, on_boundary):
                    if (dl.between(x[2], (-R_wire,R_wire+W_mag+W_SC)) and dl.between(x[1],(-R_wire-W_mag-W_SC,R_wire+W_mag+W_SC)) and ((x[0] <= 0 + 1e-3) or (x[0] >= L_wire- 1e-3)) ):
                        return True
                    else:
                        return False
                    
        elif section=='hexagonal':
            class Leads(dl.SubDomain):
                def inside(self, x, on_boundary):
                    if (dl.between(x[2], (-b0,b0+W_mag+W_SC)) and dl.between(x[1],(-R_wire-W_mag-W_SC,R_wire+W_mag+W_SC)) and ((x[0] <= 0 + 1e-3) or (x[0] >= L_wire- 1e-3)) ):
                        return True
                    else:
                        return False
                
    ##Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Substrate().mark(domains, 1)
    Wire().mark(domains, 2)
    mag_layer().mark(domains, 3)
    oxide_layer().mark(domains, 4)
    
    ##Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Back_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    Left_gate().mark(boundaries, 3)
    Right_gate().mark(boundaries, 4)
    if bc_EuS=='yes':
        SC_layers_EuS().mark(boundaries, 5)
        if bc=='open':
            Leads().mark(boundaries, 6)
    else:
        if bc=='open':
            Leads().mark(boundaries, 5)

    return (mesh),(V),(domains),(boundaries)



#%%
def Superlattice(L_wire,W_wire,W_SiO,W_SC,n_lattice,r_lattice,a_fen,W_0=0,bc='open',faces=np.array(['top']),mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire embedded in a superlattice
    array of metallic gates. Theory: S. D. Escribano et al., Phys. Rev. B 100, 
    045301 (2019).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Width of the metallic arrays.
            
        n_lattice: int
            Number of fingers distributed along the wire.
            
        r_lattice: float
            Partial coverage of the fingers with respect to the wire's length.
            The size of the fingers is therefore L_wire/n_lattice*r_lattice.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        faces: {'top','bottom'}
            Whether the mettalic gates are placed on 'top' of the wire or at its
            'bottom'.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.

            
    """
    
    #Obtain some geometrical parameters:
    a0=W_wire/2
    b0= a0*np.sin(np.pi/3.0)
    
    n_lattice=int(n_lattice)
    L_0 = float(L_wire)/n_lattice * (1 - r_lattice)
    L_SC = float(L_wire)/n_lattice * r_lattice
        
    if np.ndim(W_0)==0:
        if (bc=='open') and not(W_0==0):
            W_0=np.array([(L_SC+L_0),W_0,W_0])    
        elif (bc=='open') and (W_0==0):
            W_0=np.array([(L_SC+L_0),a0,b0])      
        else:
            W_0=np.ones(3)*a0        
    W_0x, W_0y, W_0z = W_0[0], W_0[1], W_0[2]

    ##Build the mesh if none is provided:    
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):            
            if (faces=='bottom').any():
                mesh = dl.BoxMesh(dl.Point(0,-a0-W_0y,-b0-W_SiO-W_SC),dl.Point(L_wire,a0+W_0y,b0+W_0z), int(L_wire/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))
            else:
                mesh = dl.BoxMesh(dl.Point(0,-a0-W_0y,-b0-W_SiO),dl.Point(L_wire,a0+W_0y,b0+W_0z+W_SC), int(L_wire/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))
            
            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With opern boundary conditions:
        else:        
            if (faces=='bottom').any():
                mesh = dl.BoxMesh(dl.Point(-W_0x,-a0-W_0y,-b0-W_SiO-W_SC),dl.Point(L_wire+W_0x,a0+W_0y,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))
            else:
                mesh = dl.BoxMesh(dl.Point(-W_0x,-a0-W_0y,-b0-W_SiO),dl.Point(L_wire+W_0x,a0+W_0y,b0+W_0z+W_SC), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    ##Create subdomains:
    class Dielec(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (faces=='bottom').any() and ((dl.between(x[2], (-W_SiO-b0-W_SC, -b0-W_SC)))):
                return True
            elif not((faces=='bottom').any()) and ((dl.between(x[2], (-W_SiO-b0, -b0)))):
                return True
            else:
                return False
            
    class Wire(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (dl.between(x[2], (-b0,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+a_fen*b0/a0*2)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+a_fen*b0/a0*2)) ):
                return True
            else:
                return False
    
    class Lower_gate(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (faces=='bottom').any():
                return (x[2] <= -W_SiO-b0-W_SC + a_fen)
            elif not((faces=='bottom').any()):
                return (x[2] <= -W_SiO-b0 + a_fen)
            else:
                return False
            
    if bc=='periodic':
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='top').any() and (((dl.between(x[0], (0,L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0,L_SC+L_0)))) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0-a_fen/2,b0+W_SC)))):
                    return True
                elif (faces=='bottom').any() and (((dl.between(x[0], (0,L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0,L_SC+L_0)))) and dl.between(x[1], (-a0-W_0y,a0+W_0y)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen/2)))):
                    return True
                else:
                    return False
                
    else:
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='top').any():
                    for i in range(1,n_lattice+1):
                        if (((dl.between(x[0], ((L_SC+L_0)*(i-1),(L_SC+L_0)*(i-1)+L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0+(L_SC+L_0)*(i-1),L_SC+L_0+(L_SC+L_0)*(i-1))))) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0-a_fen/2,b0+W_SC)))):
                            return True
                elif (faces=='bottom').any():
                    for i in range(n_lattice+3):
                        if (((dl.between(x[0], ((L_SC+L_0)*(i-1),(L_SC+L_0)*(i-1)+L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0+(L_SC+L_0)*(i-1),L_SC+L_0+(L_SC+L_0)*(i-1))))) and dl.between(x[1], (-a0-W_0y,a0+W_0y)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen/2)))):
                            return True
                else:
                    return False
                
    #Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    #Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    
    return (mesh),(V),(domains),(boundaries)


#%%
def Superlattice_and_Shell(L_wire,W_wire,W_SiO,W_SC,n_lattice,r_lattice,a_fen,W_0=0,Dielectric_layers='no',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire embedded in a superlattice
    array of metallic gates and partially covered by a SC shell.  Theory: S. D. 
    Escribano et al., Phys. Rev. B 100, 045301 (2019).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Width of the metallic shell and the metallic fingers.
            
        n_lattice: int
            Number of fingers distributed along the wire.
            
        r_lattice: float
            Partial coverage of the fingers with respect to the wire's length.
            The size of the fingers is therefore L_wire/n_lattice*r_lattice.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        faces: {'top','bottom'}
            Whether the mettalic gates are placed on 'top' of the wire or at its
            'bottom'.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
    """
    
    #Obtain some geometrical parameters:
    a0=W_wire/2
    b0= a0*np.sin(np.pi/3.0)
    
    n_lattice=int(n_lattice)
    L_0 = float(L_wire)/n_lattice * (1 - r_lattice)
    L_SC = float(L_wire)/n_lattice * r_lattice
    
    if np.ndim(W_0)==0:
        if (bc=='open') and not(W_0==0):
            W_0=np.array([(L_SC+L_0),W_0,W_0])    
        elif (bc=='open') and (W_0==0):
            W_0=np.array([(L_SC+L_0),a0,b0])      
        else:
            W_0=np.ones(3)*a0
    W_0x, W_0y, W_0z = W_0

    ##Build the mesh if none is provided:    
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):            
            mesh = dl.BoxMesh(dl.Point(0,-a0-W_0y,-b0-W_SiO-W_SC),dl.Point(L_wire,a0+W_0y,b0+W_0z), int(L_wire/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With open boundary conditions:
        else:        
            mesh = dl.BoxMesh(dl.Point(-W_0x,-a0-W_0y,-b0-W_SiO-W_SC),dl.Point(L_wire+W_0x,a0+W_0y,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z+W_SC)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
    
    ##Create domains:
    class Dielec(dl.SubDomain):
        def inside(self, x, on_boundary):
            if ((dl.between(x[2], (-W_SiO-b0-W_SC, -b0-W_SC)))):
                return True
            else:
                return False
            
    class Wire(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (dl.between(x[2], (-b0,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+a_fen*b0/a0*2)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+a_fen*b0/a0*2)) ):
                return True
            else:
                return False
    
    class Lower_gate(dl.SubDomain):
        def inside(self, x, on_boundary):
            return (x[2] <= -W_SiO-b0-W_SC + a_fen)

    if bc=='periodic':
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (((dl.between(x[0], (0,L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0,L_SC+L_0)))) and dl.between(x[1], (-a0-W_0y,a0+W_0y)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen/2)))):
                    return True
                else:
                    return False
                
    else:
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                for i in range(n_lattice+3):
                    if (((dl.between(x[0], ((L_SC+L_0)*(i-1),(L_SC+L_0)*(i-1)+L_SC/2)) or (dl.between(x[0], (L_SC/2+L_0+(L_SC+L_0)*(i-1),L_SC+L_0+(L_SC+L_0)*(i-1))))) and dl.between(x[1], (-a0-W_0y,a0+W_0y)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen/2)))):
                        return True
                    else:
                        return False
            
    class SC_shell(dl.SubDomain):
        def inside(self, x, on_boundary):
            if ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                return True
            else:
                return False
                
    #Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    #Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    SC_shell().mark(boundaries, 3)
    
    return (mesh),(V),(domains),(boundaries)



#%%
def Free_standing(L_wire,W_wire,W_SiO,a_fen,W_0=0,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a free-standing nanowire (deposited
    in a substrate). Experiment: S. Dhara et al., Phys. Rev. B 79, 121311(R)
    (2009).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_SiO: float
            Width of the substrate.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """

    #Obtain some geometrical parameters:
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)
    R_wire=W_wire/2

    if np.ndim(W_0)==0:
        W_0=np.ones(3)*W_wire        
    W_0x, W_0y, W_0z = W_0        

    ##Create the mesh:
    if (V==0) and (mesh==0):
        
        #With periodic boundary conditions:
        if (bc=='periodic'):
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0y,-R_wire-W_SiO),dl.Point(L_wire,R_wire+W_0y,R_wire+W_0z), int(L_wire/a_fen+1),int((W_wire+2*W_0y)/a_fen),int((W_wire+W_SiO+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-b0-W_0y,-b0-W_SiO),dl.Point(L_wire,b0+W_0y,b0+W_0z), int(L_wire/a_fen+1),int((2*b0+2*W_0y)/a_fen+1),int((2*b0+W_SiO+W_0z)/a_fen+1))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With open boundary conditions:
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-R_wire-W_0y,-R_wire-W_SiO),dl.Point(L_wire+W_0x,R_wire+W_0y,R_wire+W_0z), int((2*W_0x+L_wire)/a_fen),int((W_wire+2*W_0y)/a_fen),int((W_wire+W_SiO+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-b0-W_0y,-b0-W_SiO),dl.Point(L_wire+W_0x,b0+W_0y,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*b0+2*W_0y)/a_fen),int((2*b0+W_SiO+W_0z)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            
    ##Create domains:
    if section=='rectangular':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-R_wire, -R_wire)))):
                    return True
                else:
                    return False
        
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
        
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-R_wire + 1e-3)
                    
                
    elif section=='hexagonal':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-b0, -b0)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-b0 + 1e-3)
                
                
    #Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    #Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    
    return (mesh),(V),(domains),(boundaries)



#%%
def Free_standing2(L_wire,W_wire,W_HfO,a_fen,W_0=0,W_gate=0,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a free-standing nanowire (surrounded
    by a substrate). Experiment: K. Takase et al., Applied Physics Express 12, 
    117002 (2019).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_HfO: float
            Width of the substrate.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """
    
    #Obtain some geometrical parameters:        
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)
    R_wire=W_wire/2
    
    if W_0==0:
        W_0=W_wire/8
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):            
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_HfO,-R_wire-W_HfO-W_gate),dl.Point(L_wire,R_wire+W_0+W_HfO,R_wire+W_HfO+W_0), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(R_wire+W_HfO+W_0*0.5+W_gate*0.5)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_HfO,-b0-W_HfO-W_gate),dl.Point(L_wire,R_wire+W_0+W_HfO,b0+W_HfO+W_0), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(b0+W_HfO+W_0*0.5+W_gate*0.5)/a_fen))
                
            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With open boundary conditions:
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_HfO,-R_wire-W_HfO-W_gate),dl.Point(L_wire+W_0,R_wire+W_0+W_HfO,R_wire+W_HfO+W_0), int((L_wire+2*W_0)/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(R_wire+W_HfO+W_0*0.5+W_gate*0.5)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_HfO,-b0-W_HfO-W_gate),dl.Point(L_wire+W_0,R_wire+W_0+W_HfO,b0+W_HfO+W_0), int((L_wire+2*W_0)/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(b0+W_HfO+W_0*0.5+W_gate*0.5)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            
    ##Create the domains:
    if section=='rectangular':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and (dl.between(x[1], (-R_wire-W_HfO, -R_wire)) or dl.between(x[1], (R_wire, R_wire+W_HfO))) and dl.between(x[2], (-R_wire-W_HfO, R_wire+W_HfO))):
                    return True
                elif (dl.between(x[0], (0, L_wire)) and (dl.between(x[2], (-R_wire-W_HfO, -R_wire)) or dl.between(x[2], (R_wire, R_wire+W_HfO))) and dl.between(x[1], (-R_wire-W_HfO, R_wire+W_HfO))):
                    return True
                else:
                    return False

        class Gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (x[2] <= -R_wire-W_HfO+ 1e-3):
                    return True
                else:
                    return False
                        
    elif section=='hexagonal':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,b0)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire/2,R_wire/2)) and (dl.between(x[2],(b0,b0+W_HfO)) or dl.between(x[2],(-b0-W_HfO,-b0))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (0,b0+W_HfO)) and (dl.between(x[2],(2*b0/a0*x[1]+2*b0,2*b0/a0*x[1]+2*b0+W_HfO)) or dl.between(x[2],(-2*b0/a0*x[1]+2*b0,-2*b0/a0*x[1]+2*b0+W_HfO))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0-W_HfO,0)) and (dl.between(x[2],(-2*b0/a0*x[1]-2*b0-W_HfO,-2*b0/a0*x[1]-2*b0)) or dl.between(x[2],(2*b0/a0*x[1]-2*b0-W_HfO,2*b0/a0*x[1]-2*b0))) ):
                    return True
                else:
                    return False
                
        class Gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (x[2] <= -b0-W_HfO + 1e-3):
                    return True
                else:
                    return False
                
    ##Define the domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    ##Define the boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Gate().mark(boundaries, 1)
    
    return (mesh),(V),(domains),(boundaries)



#%%
def GAA(L_wire,W_wire,W_Al,W_Hf,a_fen,W_0=0,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a Gated-All-Around nanowire. 
    Experiment: K. Takase et al., Scientific Reports 7, 930 (2017).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_Al: float
            Width of the surounding dielectric 1.
            
        W_Hf: float
            Width of the surounding dielectric 2.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """
    
    #Obtain some geometrical paramters:        
    if section=='hexagonal':
        a0=W_wire/2
        b0=a0*np.sin(np.pi/3.0)
    R_wire=W_wire/2

    if W_0==0:
        W_0=2*a_fen
            
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_Al-W_Hf,-R_wire-W_0-W_Al-W_Hf),dl.Point(L_wire,R_wire+W_0+W_Al+W_Hf,R_wire+W_0+W_Al+W_Hf), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_Al-W_Hf,-b0-W_0-W_Al-W_Hf),dl.Point(L_wire,R_wire+W_0+W_Al+W_Hf,+b0+W_0+W_Al+W_Hf), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen+1),int(2*(+b0+W_0+W_Al+W_Hf)/a_fen+1))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
    
        #With open boundary conditions:    
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_Al-W_Hf,-R_wire-W_0-W_Al-W_Hf),dl.Point(L_wire+W_0,R_wire+W_0+W_Al+W_Hf,R_wire+W_0+W_Al+W_Hf),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_Al-W_Hf,-b0-W_0-W_Al-W_Hf),dl.Point(L_wire+W_0,R_wire+W_0+W_Al+W_Hf,+b0+W_0+W_Al+W_Hf), int((2*W_0+L_wire)/a_fen),int(2*(R_wire+W_0+W_Al+W_Hf)/a_fen+1),int(2*(+b0+W_0+W_Al+W_Hf)/a_fen+1))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            
    #Create domains:
    if section=='rectangular':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class Dielec_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and (dl.between(x[1], (-R_wire-W_Al, -R_wire)) or dl.between(x[1], (R_wire, R_wire+W_Al))) and dl.between(x[2], (-R_wire-W_Al, R_wire+W_Al))):
                    return True
                elif (dl.between(x[0], (0, L_wire)) and (dl.between(x[2], (-R_wire-W_Al, -R_wire)) or dl.between(x[2], (R_wire, R_wire+W_Al))) and dl.between(x[1], (-R_wire-W_Al, R_wire+W_Al))):
                    return True
                else:
                    return False
                
        class Dielec_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and (dl.between(x[1], (-R_wire-W_Al-W_Hf, -R_wire-W_Al)) or dl.between(x[1], (R_wire+W_Al, R_wire+W_Al+W_Hf))) and dl.between(x[2], (-R_wire-W_Al-W_Hf, R_wire+W_Al+W_Hf))):
                    return True
                elif (dl.between(x[0], (0, L_wire)) and (dl.between(x[2], (-R_wire-W_Al-W_Hf, -R_wire-W_Al)) or dl.between(x[2], (R_wire+W_Al, R_wire+W_Al+W_Hf))) and dl.between(x[1], (-R_wire-W_Al-W_Hf, R_wire+W_Al+W_Hf))):
                    return True
                else:
                    return False
        
        class Gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (abs(x[2]) >= R_wire+W_Al+W_Hf - 1e-3) or (abs(x[1]) >= R_wire+W_Al+W_Hf - 1e-3):
                    return True
                else:
                    return False
                
    elif section=='hexagonal':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,b0)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Dielec_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire/2,R_wire/2)) and (dl.between(x[2],(b0,b0+W_Al)) or dl.between(x[2],(-b0-W_Al,-b0))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (0,b0+W_Al)) and (dl.between(x[2],(2*b0/a0*x[1]+2*b0,2*b0/a0*x[1]+2*b0+W_Al)) or dl.between(x[2],(-2*b0/a0*x[1]+2*b0,-2*b0/a0*x[1]+2*b0+W_Al))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0-W_Al,0)) and (dl.between(x[2],(-2*b0/a0*x[1]-2*b0-W_Al,-2*b0/a0*x[1]-2*b0)) or dl.between(x[2],(2*b0/a0*x[1]-2*b0-W_Al,2*b0/a0*x[1]-2*b0))) ):
                    return True
                else:
                    return False
                
        class Dielec_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire/2,R_wire/2)) and (dl.between(x[2],(b0+W_Al,b0+W_Al+W_Hf)) or dl.between(x[2],(-b0-W_Al-W_Hf,-b0-W_Al))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (0,b0+W_Al+W_Hf)) and (dl.between(x[2],(2*b0/a0*x[1]+2*b0+W_Al,2*b0/a0*x[1]+2*b0+W_Al+W_Hf)) or dl.between(x[2],(-2*b0/a0*x[1]+2*b0+W_Al,-2*b0/a0*x[1]+2*b0+W_Al+W_Hf))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0-W_Al-W_Hf,0)) and (dl.between(x[2],(-2*b0/a0*x[1]-2*b0-W_Al-W_Hf,-2*b0/a0*x[1]-2*b0-W_Al)) or dl.between(x[2],(2*b0/a0*x[1]-2*b0-W_Al-W_Hf,2*b0/a0*x[1]-2*b0-W_Al))) ):
                    return True
                else:
                    return False

        class Gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and (abs(x[2])>=(b0+W_Al+W_Hf-1e-3))):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (0,b0+W_Al+W_Hf)) and ((x[2]>=2*b0/a0*x[1]+2*b0+W_Al+W_Hf-1e-3) or (x[2]>=-2*b0/a0*x[1]+2*b0+W_Al+W_Hf-1e-3)) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0-W_Al-W_Hf,0)) and ((x[2]<=-2*b0/a0*x[1]-2*b0-W_Al-W_Hf+1e-3) or (x[2]<=2*b0/a0*x[1]-2*b0-W_Al-W_Hf+1e-3)) ):
                    return True
                else:
                    return False
                
    #Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec_1().mark(domains, 1)
    Dielec_2().mark(domains, 2)
    Wire().mark(domains, 3)
    
    #Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Gate().mark(boundaries, 1)
    
    return (mesh),(V),(domains),(boundaries)



#%%
def Bigated(L_wire,W_wire,W_w_S,W_SiO,W_SC,a_fen,W_0=0,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire with a top and back gates.
    The top gate is placed far apart from the wire. Experiment: D. Liang et al.,
    Nano Lett. 12, 3263-3267 (2012).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_w_S: float
            Distance wire-substrate (for the case it is suspended somehow).
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Distance between the nanowire and the top-gate.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """
    
    #Obtain some geometrical parameters:
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)
    R_wire=W_wire/2
    
    if np.ndim(W_0)==0:
        W_0=np.ones(3)*W_wire        
    W_0x, W_0y, W_0z = W_0
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):            
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0y,-R_wire-W_SiO-W_w_S),dl.Point(L_wire,R_wire+W_0y,W_SC), int(L_wire/a_fen+1),int((W_wire+2*W_0y)/a_fen),int((R_wire+W_SiO+W_SC+W_w_S)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-b0-W_0y,-b0-W_SiO-W_w_S),dl.Point(L_wire,b0+W_0y,W_SC), int(L_wire/a_fen+1),int((2*b0+2*W_0y)/a_fen+1),int((b0+W_SiO+W_SC+W_w_S)/a_fen+1))
            
            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
          
        #With open boundary conditions:
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-R_wire-W_0y,-R_wire-W_SiO-W_w_S),dl.Point(L_wire+W_0x,R_wire+W_0y,W_SC), int((2*W_0x+L_wire)/a_fen),int((W_wire+2*W_0y)/a_fen),int((R_wire+W_SiO+W_SC+W_w_S)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-b0-W_0y,-b0-W_SiO-W_w_S),dl.Point(L_wire+W_0x,b0+W_0y,W_SC), int((2*W_0x+L_wire)/a_fen),int((2*b0+2*W_0y)/a_fen),int((b0+W_SiO+W_SC+W_w_S)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            

    ##Create the domains:
    if section=='rectangular':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-R_wire-W_w_S, -R_wire-W_w_S)))):
                    return True
                else:
                    return False

        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
        
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-R_wire + 1e-3)
                
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] >= W_SC - 1e0)
                
    elif section=='hexagonal':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_SiO-b0-W_w_S, -b0-W_w_S)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_SiO-b0 + 1e-3)
            
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] >= W_SC - 1e0)
                
    ##Define the domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    ##Define the boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    
    return (mesh),(V),(domains),(boundaries)


#%%
def Bigated2(L_wire,W_wire,W_SiO,W_HfO,a_fen,W_0=0,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire with a top and back gates.
    The top gate surrounds the wire. Experiment: I. Weperen et al., Phys. Rev. 
    B 91, 201413(R) (2015).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_SiO: float
            Width of the substrate.
            
        W_HfO: float
            Width of the dielectric between the wire and the top-gate.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        W_0: arr
            Additional distance from the nanowire center to add to the mesh in
            the three directions.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """
    
    #Obtain some geometrical parameters:    
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0) 
    R_wire=W_wire/2
    
    if W_0==0:
        W_0=2*a_fen
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_HfO,-R_wire-W_SiO),dl.Point(L_wire,R_wire+W_0+W_HfO,R_wire+W_HfO), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(R_wire+W_SiO*0.5+W_HfO*0.5)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-W_0-W_HfO,-b0-W_SiO),dl.Point(L_wire,R_wire+W_0+W_HfO,b0+W_HfO), int(L_wire/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(b0+W_SiO*0.5+W_HfO*0.5)/a_fen))
            
            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
    
        #With open boundary conditions:    
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_HfO,-R_wire-W_SiO),dl.Point(L_wire+W_0,R_wire+W_0+W_HfO,R_wire+W_HfO), int((L_wire+2*W_0)/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(R_wire+W_SiO*0.5+W_HfO*0.5)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0,-R_wire-W_0-W_HfO,-b0-W_SiO),dl.Point(L_wire+W_0,R_wire+W_0+W_HfO,b0+W_HfO), int((L_wire+2*W_0)/a_fen+1),int(2*(R_wire+W_0+W_HfO)/a_fen),int(2*(b0+W_SiO*0.5+W_HfO*0.5)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            
    ##Create the domains:
    if section=='rectangular':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
                
        class Dielec_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and (dl.between(x[1], (-R_wire-W_HfO, -R_wire)) or dl.between(x[1], (R_wire, R_wire+W_HfO))) and dl.between(x[2], (-R_wire, R_wire+W_HfO))):
                    return True
                elif (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire-W_HfO, R_wire+W_HfO)) and dl.between(x[2], (R_wire, R_wire+W_HfO))):
                    return True
                elif (dl.between(x[0], (0, L_wire)) and (dl.between(x[1], (-R_wire-W_HfO-W_0, -R_wire-W_HfO)) or dl.between(x[1], (R_wire+W_HfO, R_wire+W_HfO+W_0))) and dl.between(x[2], (-R_wire, -R_wire+W_HfO))):
                    return True
                else:
                    return False
                
        class Dielec_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-R_wire-W_SiO, -R_wire))):
                    return True
                else:
                    return False
        
        class Gate_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (abs(x[1]) >= R_wire+W_HfO - 1e-3) and (x[2]>=-R_wire+W_HfO):
                    return True
                elif (x[2]>= R_wire+W_HfO-1e-3) and (dl.between(x[1],(-R_wire-W_HfO,R_wire+W_HfO))):
                    return True
                else:
                    return False
                
        class Gate_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (x[2] <= -R_wire-W_SiO + 1e-3):
                    return True
                else:
                    return False
                
    elif section=='hexagonal':
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,b0)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Dielec_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire/2,R_wire/2)) and dl.between(x[2],(b0,b0+W_HfO)) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (0,b0+W_HfO)) and (dl.between(x[2],(2*b0/a0*x[1]+2*b0,2*b0/a0*x[1]+2*b0+W_HfO)) or dl.between(x[2],(-2*b0/a0*x[1]+2*b0,-2*b0/a0*x[1]+2*b0+W_HfO))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,0)) and (dl.between(x[1],(-a0-W_HfO,-a0/(2*b0)*x[2]-a0)) or dl.between(x[1],(a0/(2*b0)*x[2]+a0,a0+W_HfO))) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-b0+W_HfO)) and dl.between(np.abs(x[1]),(a0+W_HfO,a0+W_HfO+W_0)) ):
                    return True
                else:
                    return False
                
        class Dielec_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-W_SiO, -b0))):
                    return True
                else:
                    return False

        class Gate_1(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire/2,R_wire/2)) and (x[2]>=(b0+W_HfO)) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and (x[2]>=0) and ((x[2]>=2*b0/a0*x[1]+2*b0+W_HfO) or (x[2]>=-2*b0/a0*x[1]+2*b0+W_HfO)) ):
                    return True
                elif (dl.between(x[0], (0,L_wire)) and (x[2]>=-b0+W_HfO) and (np.abs(x[1])>=a0+W_HfO) ):
                    return True
                else:
                    return False
                
        class Gate_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (x[2] <= -b0-W_SiO + 1e-3):
                    return True
                else:
                    return False
                
    ##Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec_1().mark(domains, 1)
    Dielec_2().mark(domains, 2)
    Wire().mark(domains, 3)
    
    ##Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Gate_1().mark(boundaries, 1)
    Gate_2().mark(boundaries, 2)
    
    return (mesh),(V),(domains),(boundaries)


#%%
def Multigated(L_wire,W_wire,W_sub,sep,a_fen,section='rectangular',bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire with side-gates and a
    back gate. Experiment: Z. Schrebül et al. Phys. Rev. B 94, 035444 (2016).
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_sub: float
            Width of the substrate.
            
        sep: float
            Separation between each side-gate and the corner of the wire.
            
        a_fen: float
            Discretization of the Fenics mesh.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
    """
    
    #Obtain some geometrical parameters:
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)        
    R_wire=W_wire/2

    W_0x, W_0z = R_wire, 2*R_wire
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):                    
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-sep,-R_wire-W_sub),dl.Point(L_wire,R_wire+sep,R_wire+W_0z), int(L_wire/a_fen+1),int((W_wire+2*sep)/a_fen),int((W_wire+W_sub+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-a0-sep,-b0-W_sub),dl.Point(L_wire,a0+sep,b0+W_0z), int(L_wire/a_fen+1),int((2*a0+2*sep)/a_fen+1),int((2*b0+W_sub+W_0z)/a_fen+1))
                
            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
        
        #With open boundary conditions:
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-R_wire-sep,-R_wire-W_sub),dl.Point(L_wire+W_0x,R_wire+sep,R_wire+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*sep)/a_fen),int((W_wire+W_sub+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-a0-sep,-b0-W_sub),dl.Point(L_wire+W_0x,a0+sep,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*sep)/a_fen),int((2*b0+W_sub+W_0z)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            

    ##Create the domains:
    if section=='rectangular':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-R_wire, -R_wire)))):
                    return True
                else:
                    return False
        
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
        
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-R_wire + 1e-3)
                
                
        class Gate_L(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] <= -R_wire-sep + 1e-3) and dl.between(x[2], (-R_wire,R_wire+W_0z)))):
                    return True
                else:
                    return False
                
        class Gate_R(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] >= R_wire+sep - 1e-3) and dl.between(x[2], (-R_wire,R_wire+W_0z)))):
                    return True
                else:
                    return False
                    
    elif section=='hexagonal':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-b0, -b0)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-b0 + 1e-3)
 
                
        class Gate_L(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] <= -R_wire-sep + 1e-3) and dl.between(x[2], (-b0,b0+W_0z)))):
                    return True
                else:
                    return False
                
        class Gate_R(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] >= R_wire+sep - 1e-3) and dl.between(x[2], (-b0,b0+W_0z)))):
                    return True
                else:
                    return False
                
    ##Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    ##Define boundary conditions:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    Gate_L().mark(boundaries, 2)
    Gate_R().mark(boundaries,3)
    
    return (mesh),(V),(domains),(boundaries)



#%%
def Multigated_and_shell(L_wire,W_wire,W_sub,W_SC,a_fen,sep,section='rectangular',faces=np.array(['1']),bc='open',mesh=0,V=0):
    """
    Build the electrostatic environment of a nanowire covered with a metallic
    shell and surrounded by three gates, two side-gates and one back-gate.
    Experiment: Majorana's experiments.
    
    Parameters
    ----------
        L_wire: float
            Length of the wire.
        
        W_wire: float
            Width of the wire.
            
        W_sub: float
            Width of the substrate.
            
        W_SC: float
            Width of the metallic layer.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        sep: float
            Separation between each side-gate and the corner of the wire.
            
        section: {"rectangular","hexagonal"}
            Section profile of the nanowire.
            
        faces: arr
            Facets that the metallic shell covers the wire. Each facet is
            labeled with a number from 1 to 6 (the upper one is 1, and the
            rest are numbered clockwise). Each element of the array denotes
            with a string (e.g. np.array(['1','2'])) if such facet is covered.
            
        bc: {"open","periodic"}
            Boundary conditions of the wire.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
    """
    
    #Obtain some geometrical parameters:    
    if section=='hexagonal':
        a0=W_wire/2
        b0= a0*np.sin(np.pi/3.0)        
    R_wire=W_wire/2
    
    W_0x, W_0z = R_wire, 3*W_SC
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        #With periodic boundary conditions:
        if (bc=='periodic'):                    
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(0,-R_wire-sep,-R_wire-W_sub),dl.Point(L_wire,R_wire+sep,R_wire+W_0z), int(L_wire/a_fen+1),int((W_wire+2*sep)/a_fen),int((W_wire+W_sub+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(0,-a0-sep,-b0-W_sub),dl.Point(L_wire,a0+sep,b0+W_0z), int(L_wire/a_fen+1),int((2*a0+2*sep)/a_fen+1),int((2*b0+W_sub+W_0z)/a_fen+1))

            #Impose periodicity:
            class PeriodicBoundary(dl.SubDomain):
                def inside(self, x, on_boundary):
                    return bool(x[0] < dl.DOLFIN_EPS and x[0] > -dl.DOLFIN_EPS and on_boundary)
                def map(self, x, y):
                    y[0] = x[0] - L_wire
                    y[1] = x[1]
                    y[2] = x[2]
            pbc = PeriodicBoundary()
            V = dl.FunctionSpace(mesh, 'Lagrange', 1, constrained_domain=pbc)
            
        #With open boundary conditions:
        else:
            if section=='rectangular':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-R_wire-sep,-R_wire-W_sub),dl.Point(L_wire+W_0x,R_wire+sep,R_wire+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*sep)/a_fen),int((W_wire+W_sub+W_0z)/a_fen))
            elif section=='hexagonal':
                mesh = dl.BoxMesh(dl.Point(-W_0x,-a0-sep,-b0-W_sub),dl.Point(L_wire+W_0x,a0+sep,b0+W_0z), int((2*W_0x+L_wire)/a_fen),int((2*a0+2*sep)/a_fen),int((2*b0+W_sub+W_0z)/a_fen))
            V = dl.FunctionSpace(mesh, 'Lagrange', 1)
            

    ##Create domains:
    if section=='rectangular':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-R_wire, -R_wire)))):
                    return True
                else:
                    return False

        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (0, L_wire)) and dl.between(x[1], (-R_wire, R_wire)) and dl.between(x[2], (-R_wire, R_wire))):
                    return True
                else:
                    return False
        
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-R_wire + 1e-3)
                
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='2').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (R_wire,R_wire+W_SC)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='4').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire-W_SC,-R_wire)) and dl.between(x[2], (-R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='1').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (R_wire,R_wire+W_SC)))):
                    return True
                elif (faces=='3').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-R_wire,R_wire)) and dl.between(x[2], (-R_wire-W_SC,-R_wire)))):
                    return True
                else:
                    return False
                
        class Gate_L(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] <= -R_wire-sep + 1e-3) and dl.between(x[2], (-R_wire,R_wire+W_0z)))):
                    return True
                else:
                    return False
                
        class Gate_R(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] >= R_wire+sep - 1e-3) and dl.between(x[2], (-R_wire,R_wire+W_0z)))):
                    return True
                else:
                    return False
                
    elif section=='hexagonal':
        class Dielec(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[2], (-W_sub-b0, -b0)))):
                    return True
                else:
                    return False
                
        class Wire(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[2], (-b0-a_fen,b0)) and dl.between(x[0], (0,L_wire)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                    return True
                else:
                    return False
                
        class Lower_gate(dl.SubDomain):
            def inside(self, x, on_boundary):
                return (x[2] <= -W_sub-b0 + 1e-3)
            
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (faces=='1').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (b0,b0+W_SC)))):
                    return True                    
                elif (faces=='2').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                    return True
                elif (faces=='6').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                    return True
                elif (faces=='3').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_SC,-2*b0/a0*x[1]+2*b0)))):
                    return True
                elif (faces=='5').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                    return True
                elif (faces=='4').any() and ((dl.between(x[0], (0,L_wire)) and dl.between(x[1], (-a0/2,a0/2)) and dl.between(x[2], (-b0-W_SC,-b0+a_fen)))):
                    return True
                else:
                    return False
                
        class Gate_L(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] <= -R_wire-sep + 1e-3) and dl.between(x[2], (-b0,b0+W_0z)))):
                    return True
                else:
                    return False
                
        class Gate_R(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (0,L_wire)) and (x[1] >= R_wire+sep - 1e-3) and dl.between(x[2], (-b0,b0+W_0z)))):
                    return True
                else:
                    return False
                                
    ##Define domains:
    domains = dl.MeshFunction("size_t", mesh,3)
    domains.set_all(0)
    Dielec().mark(domains, 1)
    Wire().mark(domains, 2)
    
    ##Define boundaries:
    boundaries = dl.MeshFunction("size_t", mesh,2)
    boundaries.set_all(0)
    Lower_gate().mark(boundaries, 1)
    SC_layers().mark(boundaries, 2)
    Gate_L().mark(boundaries, 3)
    Gate_R().mark(boundaries,4)
    
    return (mesh),(V),(domains),(boundaries)


#%%
def JJ_2DEG(L_N,L_SC,Y_wire,Z_wire,W_SiO,W_SC,a_fen,L_gate=0,x_0=0,SC2='no',mesh=0,V=0):
    """
    Build the electrostatic environment for a Josephson junction made of a 2DEG.
    Theory: A. A. Reynoso et al., Phys. Rev. B 86, 214519 (2012).
    
    Parameters
    ----------
        L_N: float
            Length of the junction.
        
        L_SC: float
            Length of the superconductors to include in the simulations.
            
        Y_wire: float
            width of the junction.
            
        Z_wire: float
            Width of the 2DEG.
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Width of the superconductors.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        L_gate: float
            Length of the lateral gate.
            
        x_0: float
            Offset of the lateral gate position with respect to the middle of
            the junction.
            
        SC2: {"yes","no"}
            Whether both superconductors at both ends of the wire have the same
            electrostatic boundary condition.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
    """
    
    #Obtain some geometrical parameters:
    W_0z=W_SC
    if L_gate==0:
        L_gate=L_N
        
    ##Cretae the mesh
    if (V==0) and (mesh==0):
        mesh = dl.BoxMesh(dl.Point(-L_SC,0,-W_SiO-W_SC-Z_wire/2),dl.Point(L_N+L_SC,Y_wire,Z_wire/2+W_0z), int((2*L_SC+L_N)/a_fen),int((Y_wire)/a_fen),int((W_SiO+Z_wire+W_SC+W_0z)/a_fen))
        V = dl.FunctionSpace(mesh, 'Lagrange', 1)
        
    ##Create domains:
    class Dielec(dl.SubDomain):
        def inside(self, x, on_boundary):
            if ((dl.between(x[2], (-W_SiO-W_SC-Z_wire/2, -W_SC-Z_wire/2)))):
                return True
            else:
                return False
            
    class Wire(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (dl.between(x[0], (-L_SC,L_N+L_SC)) and dl.between(x[2], (-Z_wire/2,Z_wire/2)) ):
                return True
            else:
                return False
            
    class Lower_gate(dl.SubDomain):
        def inside(self, x, on_boundary):
            return ((x[2] <= -W_SiO-W_SC-Z_wire/2 + 1e-3) and (dl.between(x[0],(0+x_0,L_gate+x_0))))
    
    
    if SC2=='no':
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (-L_SC,0)) or dl.between(x[0], (L_N,L_N+L_SC))) and (dl.between(x[2],(-W_SC-Z_wire/2,-Z_wire/2)) or dl.between(x[2],(Z_wire/2,Z_wire/2+W_SC)))):
                    return True
                else:
                    return False
                    
        #Define domains:
        domains = dl.MeshFunction("size_t", mesh,3)
        domains.set_all(0)
        Dielec().mark(domains, 1)
        Wire().mark(domains, 2)
        
        #Define boundaries:
        boundaries = dl.MeshFunction("size_t", mesh,2)
        boundaries.set_all(0)
        Lower_gate().mark(boundaries, 1)
        SC_layers().mark(boundaries, 2)
    
    
    else:
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (L_N,L_N+L_SC))) and (dl.between(x[2],(-W_SC-Z_wire/2,-Z_wire/2)) or dl.between(x[2],(Z_wire/2,Z_wire/2+W_SC)))):
                    return True
                else:
                    return False
                
        class SC_layers_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (-L_SC,0))) and (dl.between(x[2],(-W_SC-Z_wire/2,-Z_wire/2)) or dl.between(x[2],(Z_wire/2,Z_wire/2+W_SC)))):
                    return True
                else:
                    return False

        #Define domains:            
        domains = dl.MeshFunction("size_t", mesh,3)
        domains.set_all(0)
        Dielec().mark(domains, 1)
        Wire().mark(domains, 2)
        
        #Define boundaries:
        boundaries = dl.MeshFunction("size_t", mesh,2)
        boundaries.set_all(0)
        Lower_gate().mark(boundaries, 1)
        SC_layers().mark(boundaries, 2)
        SC_layers_2().mark(boundaries, 3)
        
    
    return (mesh),(V),(domains),(boundaries)



#%%
def JJ_nanowire(L_N,L_SC,Y_wire,Z_wire,W_SiO,W_SC,a_fen,L_gate=0,x_0=0,SC2='no',mesh=0,V=0):
    """
    Build the electrostatic environment for a Josephson junction made of a
    (hexagonal cross-section) nanowire. Experiment: L. Tosi et al., Phys. Rev.
    X 9, 011010 (2019).
    
    Parameters
    ----------
        L_N: float
            Length of the junction.
        
        L_SC: float
            Length of the superconductors to include in the simulations.
            
        Y_wire: float
            width of the junction.
            
        Z_wire: float
            Width of the 2DEG.
            
        W_SiO: float
            Width of the substrate.
            
        W_SC: float
            Width of the superconductors.
        
        a_fen: float
            Discretization of the Fenics mesh.
            
        L_gate: float
            Length of the lateral gate.
            
        x_0: float
            Offset of the lateral gate position with respect to the middle of
            the junction.
            
        SC2: {"yes","no"}
            Whether both superconductors at both ends of the wire have the same
            electrostatic boundary condition.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        V: Fenics function
            Fenics space function to be reused.            
            
            
    Returns
    -------
        mesh: Fenics mesh
            Fenics mesh of the environment.

        V: Fenics function
            Fenics space function with the environment.
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
    """
    
    #Obtain some geometrical parameters:         
    a0=Y_wire/2
    b0= a0*np.sin(np.pi/3.0)  
    
    W_0z=W_SC
    W_0y, W_0z= Y_wire/2, Z_wire/2    
    
    if L_gate==0:
        L_gate=L_N

    if np.isscalar(a_fen):
        a_fen=np.ones(3)*a_fen
    
    ##Create the mesh:
    if (V==0) and (mesh==0):
        mesh = dl.BoxMesh(dl.Point(-L_SC,-a0-W_0y,-b0-W_SiO-W_SC),dl.Point(L_N+L_SC,a0+W_0y,b0+W_0z), int((L_N+2*L_SC)/a_fen[0]),int((2*a0+2*W_0y)/a_fen[1]),int((2*b0+W_SiO+W_0z+W_SC)/a_fen[2]))
        V = dl.FunctionSpace(mesh, 'Lagrange', 1)

    ##Create domains:
    class Dielec(dl.SubDomain):
        def inside(self, x, on_boundary):
            if ((dl.between(x[2], (-W_SiO-W_SC-b0, -W_SC-b0)))):
                return True
            else:
                return False
            
    class Wire(dl.SubDomain):
        def inside(self, x, on_boundary):
            if (dl.between(x[2], (-b0-a_fen[2],b0)) and dl.between(x[0], (-L_SC,L_N+L_SC)) and dl.between(x[2],(2*b0/a0*x[1]-2*b0-a_fen[2]*b0/a0*2,-2*b0/a0*x[1]+2*b0)) and dl.between(x[2],(-2*b0/a0*x[1]-2*b0-a_fen[2]*b0/a0*2,2*b0/a0*x[1]+2*b0)) ):
                return True
            else:
                return False
                            
    class Lower_gate(dl.SubDomain):
        def inside(self, x, on_boundary):
            return ((x[1] <= -a0-W_0y + 3*a_fen[1]) and (dl.between(x[0],(0+x_0,L_gate+x_0))))
        
    
    if SC2=='no':
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (-L_SC,0)) or (dl.between(x[0], (L_N,L_N+L_SC)))):
                    if (dl.between(x[1], (-a0/2,a0/2)) and (dl.between(x[2], (b0-a_fen[2],b0+W_SC)) or dl.between(x[2], (-b0-W_SC,-b0+a_fen[2])))):
                        return True
                    elif ((dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_SC,-2*b0/a0*x[1]+2*b0)))):
                        return True
                    elif ((dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                        return True
                    else:
                        return False
                else:
                    return False
                    
        #Define domains:
        domains = dl.MeshFunction("size_t", mesh,3)
        domains.set_all(0)
        Dielec().mark(domains, 1)
        Wire().mark(domains, 2)
        
        #Define boundaries:
        boundaries = dl.MeshFunction("size_t", mesh,2)
        boundaries.set_all(0)
        Lower_gate().mark(boundaries, 1)
        SC_layers().mark(boundaries, 2)
        
        
    else:
        class SC_layers(dl.SubDomain):
            def inside(self, x, on_boundary):
                if ((dl.between(x[0], (L_N,L_N+L_SC)))):
                    if (dl.between(x[1], (-a0/2,a0/2)) and (dl.between(x[2], (b0-a_fen[2],b0+W_SC)) or dl.between(x[2], (-b0-W_SC,-b0+a_fen[2])))):
                        return True
                    elif ((dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_SC,-2*b0/a0*x[1]+2*b0)))):
                        return True
                    elif ((dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                        return True
                    else:
                        return False
                else:
                    return False
            
        class SC_layers_2(dl.SubDomain):
            def inside(self, x, on_boundary):
                if (dl.between(x[0], (-L_SC,0))):
                    if (dl.between(x[1], (-a0/2,a0/2)) and (dl.between(x[2], (b0-a_fen[2],b0+W_SC)) or dl.between(x[2], (-b0-W_SC,-b0+a_fen[2])))):
                        return True
                    elif ((dl.between(x[2], (-2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0,-2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (2*b0/a0*x[1]+2*b0-a_fen[2]*b0/a0*2,-2*b0/a0*x[1]+W_SC)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0,2*b0/a0*x[1]+2*b0+W_SC)))):
                        return True
                    elif ((dl.between(x[2], (-b0,2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (2*b0/a0*x[1]-2*b0-W_SC,-2*b0/a0*x[1]+2*b0)))):
                        return True
                    elif ((dl.between(x[2], (-b0,-2*b0/a0*x[1]-2*b0+a_fen[2]*b0/a0*2)) and dl.between(x[2], (-2*b0/a0*x[1]-2*b0-W_SC,2*b0/a0*x[1]+2*b0)))):
                        return True
                    else:
                        return False
                else:
                    return False
                    
        #Define domains:
        domains = dl.MeshFunction("size_t", mesh,3)
        domains.set_all(0)
        Dielec().mark(domains, 1)
        Wire().mark(domains, 2)
        
        #Define boundaries:
        boundaries = dl.MeshFunction("size_t", mesh,2)
        boundaries.set_all(0)
        Lower_gate().mark(boundaries, 1)
        SC_layers().mark(boundaries, 2)
        SC_layers_2().mark(boundaries, 3)
        
    return (mesh),(V),(domains),(boundaries)


#%%############################################################################
######################    Environment solver      ###########################
###############################################################################

#%%
def Density_Fenics(den,points_site,points_mesh,mesh,V):
    """
    Transform the charge density written in a numpy array to a function in 
    Fenics.
    
    Parameters
    ----------
        den: arr
            Charge density on each site of the wire.
        
        points_site: arr
            Array encoding the points where each charge density point of den is
            located in space.
            
        points_site: arr
            Array enconding the points where the it is desired to locate the
            charge density in the Fenics mesh. Therefore, it must have the same
            boundaries than points_site but with a different discretization (or
            not).
            
        mesh: Fenics mesh
            Fenics mesh for the system.
            
        V: Fenics function
            Fenics space function of the mesh.
            
            
    Returns
    -------
        den_Fenics: function
            A function with the charge density in the Fenics space.
            
    """
    
    if not(np.abs(np.sum(points_site[0,:]-points_site[1,:]))==np.abs(np.sum(points_mesh[0,:]-points_mesh[1,:]))) :
        den_poisson = interpolate.griddata(points_site, den, points_mesh, method='linear')
        den_poisson[np.isnan(den_poisson)] = 0
    else:
        den_poisson=den
        points_mesh=points_site
 
    to_1d = lambda x: x.view(dtype=np.dtype([('f1', 'float'),('f2', 'float'),('f3', 'float')]))[:, 0]
    n,d = V.dim(),mesh.geometry().dim()
    V_dof_coordinates = V.tabulate_dof_coordinates()
    V_dof_coordinates.resize((n,d))
    ordering = np.argsort(to_1d(V_dof_coordinates))
    wire_indices = np.searchsorted(to_1d(V_dof_coordinates),
                                   to_1d(points_mesh), sorter=ordering)
    den_Fenics = dl.Function(V)
    den_Fenics.vector()[ordering[wire_indices]] = den_poisson
    
    return den_Fenics


#%%
def Solve_Poisson(V,domains,boundaries,ep,vol,sol=0,mesh=0,den=0,x=0,y=0,z=0):
    """
    Solve the poisson equation. It returns the electrostatic potential in 
    the Fenics space.
    
    Parameters
    ----------
        V: Fenics function
            Fenics space function to be reused.            
            
        domains: Fenics function
            Fenics space function with the domains.
            
        boundaries: Fenics function
            Fenics space function with the boundaries.
            
        ep: array or scalar
            Dielectric constant inside each material. Same ordering than
            domains.
            
        vol: array or scalar
            Electrostatic potential to be fixed on each boundary. Same ordering
            than boundaries.
            
        sol: Fenics function
            Initial guess for the iterative process.
            
        mesh: Fenics mesh
            Fenics mesh to be reused.
            
        den: array
            Source term (charge density) in a given arbitrary mesh (x,y,z)
            inside the nanowire.
            
        x,y,z: arrays
            Arrays describing the spatial points where the charge density den 
            is located.
            
            
    Returns
    -------
        u: Fenics function
            Fenics function with the solution to the Poisson equation. It
            corresponds to the elecgtrostatic potential evaluated at any point 
            inside the mesh.
            
    """
    
    #Obtain some geometrical parameters:
    Nx, Ny, Nz = len(x), len(y), len(z)
    ep = ep*cons.epsilon_0*10**(-12)/cons.e
        
    ##Create the source term:
    if np.sum(den)==0:
        f=dl.Constant(0)
        
    else:                
        #Obtain the points in the fenics mesh:
        mesh_points_all = mesh.coordinates() 
        length = len(mesh_points_all)   
        
        mesh_points_filter_x, mesh_points_filter_y, mesh_points_filter_z = [], [], []
        for i in range(length):
            if (mesh_points_all[i,0] <=x[Nx-1]+ 1e-1)and(mesh_points_all[i,0] >=x[0]- 1e-1)and(abs(mesh_points_all[i,1]) <=y[Ny-1]+ 1e-1)and(abs(mesh_points_all[i,2]) <=z[Nz-1]+ 1e-1):
                mesh_points_filter_x.append(mesh_points_all[i,0])
                mesh_points_filter_y.append(mesh_points_all[i,1])
                mesh_points_filter_z.append(mesh_points_all[i,2])
        
        N_fen=len(mesh_points_filter_x)
        points_mesh=np.zeros([N_fen,3])
        for i in range(N_fen):
            points_mesh[i,:]=np.array([mesh_points_filter_x[i],mesh_points_filter_y[i],mesh_points_filter_z[i]])
    
        #Obtain the points in the tight-binding mesh:
        points_site, den_site = np.zeros([Nx*Ny*Nz,3]), np.zeros([Nx*Ny*Nz])
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    points_site[i+(j*Nz+k)*Nx,:]=np.array([x[i],y[j],z[k]])
                    den_site[i+(j*Nz+k)*Nx]=den[i,j,k]    
        
        #Obtain the charge density in the fenics mesh (which is the source term):
        f=Density_Fenics(den_site*(-1),points_site,points_mesh,mesh,V)   
    
    ##Build the Poisson equation:
    dx = dl.Measure("dx")(subdomain_data=domains)
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)
    
    #Left part of the functional:
    if np.isscalar(ep):
        a = dl.inner(ep*dl.grad(u), dl.grad(v))*dx(0)
    else:
        a=0
        for i in range(len(ep)):
            a += dl.inner(float(ep[i])*dl.grad(u), dl.grad(v))*dx(i)
    
    #Right part of the functional:
    L = f*v*dx(2)
    
    #Boundary conditions:
    if np.isscalar(vol):
        bcs = [dl.DirichletBC(V, vol, boundaries, 1)]
    else:
        bcs = []
        for i in range(len(vol)):
            bcs.append(dl.DirichletBC(V, vol[i], boundaries, i+1))

    ##Solve the equation:
    if np.sum(sol)==0:
        u = dl.Function(V)
        dl.solve(a == L, u, bcs, solver_parameters={'linear_solver':'cg','preconditioner': 'ilu'})
    else:
        u=sol
        dl.parameters["krylov_solver"]["nonzero_initial_guess"]= True
        dl.solve(a == L, u, bcs, solver_parameters={'linear_solver':'cg','preconditioner': 'ilu'})
        
    return (u)

