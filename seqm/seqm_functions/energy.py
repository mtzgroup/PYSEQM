import torch
from .constants import a0
import time


def elec_energy_isolated_atom(const, Z, uss, upp, gss, gpp, gsp, gp2, hsp):
    """
    electrionc energy for a single atom
    #eisol in block.f or in calpar.f
    return Eiso, shape (natoms,)
    """
    Eiso = uss*const.ussc[Z] \
          + upp*const.uppc[Z] \
          + gss*const.gssc[Z] \
          + gpp*const.gppc[Z] \
          + gsp*const.gspc[Z] \
          + gp2*const.gp2c[Z] \
          + hsp*const.hspc[Z]
    return Eiso

def elec_energy(P,F,Hcore):
    """
    Get the electronic energy
    P: density matrix, shape (nmol, molsize*4, molsize*4)
    F: fock matrix, shape same as P
    Hcore: Hcore matrix, shape (nmol, 4*molsize, 4*molsize)
    return Eelec : electronic energy, shape (nmol,)
    P, F: full, has upper and lower triangle
    Hcore : only have upper triangle as constructed from hcore.py
    """
    h = Hcore.triu()+Hcore.triu(1).transpose(1,2)

    # Eelec = 0.5 * tr(P(Hcore+F))  # matmul
    # Eelec = 0.5 * \sum P*(H+F)    # elementwise product
    
    if len(F.size()) == 4:
        Eelec = 0.5*torch.sum((P[:,0]+P[:,1])*h+P[:,0]*F[:,0] + P[:,1]*F[:,1],dim=(1,2))
    else:
        Eelec = 0.5*torch.sum(P*(h+F),dim=(1,2))
    # print('h\n', h)
    # print('P\n', P)
    # print('F\n', F)
    # print('Eelec\n', Eelec)

    return Eelec

# def elec_energy_open_shell(P,F,Hcore):
#     """
#     Get the electronic energy
#     P: density matrix, shape (nmol, molsize*4, molsize*4)
#     F: fock matrix, shape same as P
#     Hcore: Hcore matrix, shape (nmol, 4*molsize, 4*molsize)
#     return Eelec : electronic energy, shape (nmol,)
#     P, F: full, has upper and lower triangle
#     Hcore : only have upper triangle as constructed from hcore.py
#     """
#     h = Hcore.triu()+Hcore.triu(1).transpose(1,2)

#     # Eelec = 0.5 * tr(P(Hcore+F))  # matmul
#     # Eelec = 0.5 * \sum P*(H+F)    # elementwise product

#     Eelec = 0.5*torch.sum(P*(h+F),dim=(1,2))
#     # print('h\n', h)
#     # print('P\n', P)
#     # print('F\n', F)
#     # print('Eelec\n', Eelec)

#     return Eelec


def elec_energy_xl(D,P,F,Hcore):
    """
    XL_BOMD
    electrionic energy is defined as:
    E(D,P) = (2*tr(Hcore*D) + tr((2D-P)*G(P)))/2.0
           = tr(D*F)-0.5*Tr((F-Hcore)*P)
    """
    #Hcore : only have upper triangle as constructed from hcore.py
    h = Hcore.triu()+Hcore.triu(1).transpose(1,2)

    Eelec = torch.sum(D*F-0.5*(F-h)*P,dim=(1,2))

    return Eelec


def pair_nuclear_energy(const, nmol, ni, nj, idxi, idxj, rij, gam, method='AM1', parameters=None):
    """
    Compute Nuclear Energy
    method='MNDO', 'AM1', 'PM3'
    nmol : number of molecules
    pair_molid : molecule id for each pair, shape (npairs,)
    ni, nj: atomic number, shape (npairs,)
    rij: pair distance in atomic units, shape (npairs,)
    gam : (s^A s^A, s^B, s^B) = w[...,0,0], shape(npairs,): w ==> second return vaule of hcore
    parameters : tuple, (alpha,) or (alpha, K, L, M)
        alpha : shape (natoms,) or (natoms, natoms)
        chi:    shape (natoms, natoms) for AM1_PDREP
        K,L,M : guassian terms in PM3 or AM1, shape (natoms, 2 or 4)
    return nuclear interaction energy for each molecule, (nmol, )
    """
    rija = rij * a0
    tore = const.tore
    ## t1 = ZA * ZB * <sA sA|sB sB>
    t1 = tore[ni] * tore[nj] * gam  # tore = valence charge, not Z!
    
    if method == 'AM1_PDREP':
        """
        EnucAB = Vusr_AB + ZA * ZB * <sA sA|sB sB> * (1 + chi_AB * exp(-alpha_AB * R_AB))
        """
        alpha, chi, K, L, M = parameters
        
        # ultra short-range potential ("unpolarizable core-core") = 1e-8 * ((Z_A^1/3 + Z_B^1/3) / R_AB)^12
        Vusr = 1e-8 * torch.pow( (torch.pow(ni, 1/3) + torch.pow(nj, 1/3)) / rija, 12)
        
        t2 = chi[idxi,idxj] * torch.exp(-alpha[idxi,idxj] * rija)
#        t3_1 = tore[ni] * tore[nj] / rija
#        t3_2 = torch.sum(K[idxi] * torch.exp(-L[idxi] * (rija.reshape((-1,1)) - M[idxi])**2), dim=1)
#        t3_3 = torch.sum(K[idxj] * torch.exp(-L[idxj] * (rija.reshape((-1,1)) - M[idxj])**2), dim=1)
#        t3 = t3_1 * (t3_2 + t3_3)
        t3 = 0.
        EnucAB = Vusr + t1 * (1.0 + t2) + t3
        return EnucAB
    else:
        alpha = parameters[0]
        # special cases for N-H and O-H
        XH = ((ni==7) | (ni==8)) & (nj==1)
        ## A = N or O and B = H: t2 = R_AB * exp(-alpha_A * R_AB)
        ##                 else: t2 = exp(-alpha_A * R_AB)
        t2 = torch.zeros_like(t1)
        tmp = torch.exp(-alpha[idxi] * rija)
        t2[~XH] = tmp[~XH]
        t2[XH] = tmp[XH] * rija[XH]
        ## t3 = exp(-alpha_B * R_AB)
        t3 = torch.exp(-alpha[idxj] * rija)
    if method=='MNDO':
        #in mopac, rij is in unit of angstrom
        # EnucAB = ZA * ZB * <sA sA|sB sB> * (1 + t2 + exp(-alpha_B * R_AB))
        EnucAB = t1 * (1.0 + t2 + t3)
    elif method=='PM3' or method=='AM1':
        #two gaussian terms for PM3
        # 3~4 terms for AM1
        _, K, L, M = parameters
        #K, L , M shape (natoms,2 or 4)
        t4 = tore[ni]*tore[nj]/rija
        t5 = torch.sum(K[idxi]*torch.exp(-L[idxi]*(rija.reshape((-1,1))-M[idxi])**2),dim=1)
        t6 = torch.sum(K[idxj]*torch.exp(-L[idxj]*(rija.reshape((-1,1))-M[idxj])**2),dim=1)
        ## EnucAB = EnucAB[MNDO] + Z_A * A_B / R_AB * sum K_i exp(-L_i * (R_AB - M_i)^2)
        EnucAB = t1 * (1.0 + t2 + t3) + t4 * (t5 + t6)
    else:
        raise ValueError("Supported Methods: MNDO, AM1, PM3, AM1_PDREP")
    return EnucAB

def total_energy(nmol, pair_molid, EnucAB, Eelec):
    """
    total energy for each molecule
    total energy E_tot^mol= Eelec + \sum{pair A,B,A<B} E_nuc^AB
    #nuclear energy between pair of atom A and B: E_nuc^AB

    as index_add is expensive, there is no need to do this during training
    EnucAB :computed from pair_nuclear_energy, shape (npairs,)
    pair_molid : molecule id for each pair
    Eelec : electronic energy for each molecule, computed from elec_energy, shape (nmol)

    """
    Enuc = torch.zeros((nmol,), dtype=EnucAB.dtype, device=EnucAB.device)
    Enuc.index_add_(0, pair_molid, EnucAB)
    Etot = Eelec + Enuc
    return Etot, Enuc

def heat_formation(const, nmol, atom_molid, Z, Etot, Eiso, flag=True):
    """
    get the heat of formation for each molecule
    return Hf : shape (nmol,)
    #heat of formation : delta H_f^mol
    #electronic energies of isolated atom: E_el^A
    #experimental heat of formation of isolatied atom : delta_H_f^A
    # delta H_f^mol = E_tot^mol - \sum_A E_el^A + \sum_A delta_H_f^A
    #flag: True, return Hf = Etot - Eiso_sum + eheat_sum
           False, return Etot - Eiso_sum
    """
    # electronic energy for isolated atom, sum for each molecule
    Eiso_sum = torch.zeros_like(Etot)
    Eiso_sum.index_add_(0,atom_molid,Eiso)
    if flag:
        # experimental heat of formation for each atom, sum for each molecule
        eheat_sum = torch.zeros_like(Etot)
        eheat_sum.index_add_(0,atom_molid,const.eheat[Z])
        #Hf = Etot - Eiso_sum + eheat_sum
        return Etot - Eiso_sum + eheat_sum, Eiso_sum
    else:
        return Etot - Eiso_sum, Eiso_sum
