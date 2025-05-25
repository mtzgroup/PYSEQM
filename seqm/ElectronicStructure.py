import torch
from torch.autograd import grad as adgrad
from torch.nested import as_nested_tensor
from .basics import *
from seqm.XLBOMD import ForceXL
#from seqm.XLBOMD_LR import ForceXL as ForceXL_lr



def eckart(masses, coordinates):
    M_mol = masses.sum(dim=(1,2))
    COMs = (masses * coordinates).sum(dim=1) / M_mol
    R0 = coordinates - COMs
    I = torch.einsum("bij,bik->bjk", masses * R0, R0)
    _, X = torch.linalg.eigh(I / M_mol)
    R_Eckart = R0.bmm(X)
    return X, R_Eckart

def vibrational_basis(masses, coordinates):
    # get Eckart frame
    X, R_Eck = eckart(masses, coordinates)
    # Eigenbasis for translations and rotations
    T = torch.eye(3).repeat(*coordinates.shape[:2], 1)
    G_Eck = R_Eck * masses.sqrt()
    R = torch.stack([torch.cat([torch.stack([gij.cross(xi[i]) for i in range(3)])
                     for gij in gi]) for (gi, xi) in zip(G_Eck, X)])
    TR = torch.cat((T, R), 2)
    # Eigenspace of global TR = net translations and rotations
    # -> remove subspace corresponding to the 6 (5) non-zero
    #    singular values for non-linear (linear) molecules
    U, S, _ = torch.linalg.svd(TR, full_matrices=True)
    nn0 = S.round_(decimals=15).count_nonzero(dim=-1)
    B = as_nested_tensor([Ui[:,n:] for (Ui,n) in zip(U, nn0)])
    return B

def tuple2nested(x): return as_nested_tensor(list(x))

def harmonic_analysis(masses, coordinates, hessian):
    im = masses.pow(-0.5).repeat_interleave(3, dim=1).squeeze(-1)
#    Hm = torch.einsum('bi,bj->bij', (m, m)) * hessian
    Hm = as_nested_tensor([im_i.unsqueeze(-1) * Hi * im_i for (im_i,Hi) in zip(im, hessian)])
    B = vibrational_basis(masses, coordinates)
    BTHB = B.transpose(-1,-2).bmm(Hm.bmm(B))
#    e, v = torch.linalg.eigh(BTHB)
    # update once `eigh` supported by nested
    e, v = map(tuple2nested, zip(*map(torch.linalg.eigh, BTHB)))
    U = B.bmm(v)
    M = (im_i.unsqueeze(-1) for im_i in im)
    Q = nestedmap_binary(torch.mul, U, M)
#    Q = U * torch.einsum('bi,bj->bij', (im, torch.ones_like(U[:,0,:])))
    return e, Q.transpose(-1,-2)

def nestedmap_unary(f, x):
    ## update once complex supported in `as_nested_tensor`
    return torch.nested.nested_tensor(list(map(f, x)))

def nestedmap_binary(f, x, y):
    ## update once complex supported in `as_nested_tensor`
    return torch.nested.nested_tensor(list(map(f, x, y)))

def curve2freq(x):
    # update once `type` and `sqrt` supported by nested
    freq = x.type(torch.complex64).sqrt()
    ## ignore imaginary freqencies small than 1 cm^-1
    if (freq.imag.abs() < 1.9176525463471034e-3).all(): freq = freq.real
    return freq * 0.06465415105180661 ## curvature to frequency


class Electronic_Structure(torch.nn.Module):
    def __init__(self, seqm_parameters, *args, **kwargs):
        """
        unit for timestep is femtosecond
        output: [molecule id list, frequency N, prefix]
            molecule id in the list are output, staring from 0 to nmol-1
            geometry is writted every dump step to the file with name prefix + molid + .xyz
            step, temp, and total energy is print to screens for select molecules every thermo
        """
        super().__init__(*args, **kwargs)
        #self.molecule = molecule
        self.seqm_parameters = seqm_parameters
        self.conservative_force = Force(self.seqm_parameters)
        self.conservative_force_xl = ForceXL(self.seqm_parameters)
        #self.conservative_force_xl_lr = ForceXL_lr(self.seqm_parameters)
        self.do_hessian = seqm_parameters.get("hessian", False)
        self.keep_graph = seqm_parameters.get("2nd_grad", False)
        
        #self.acc_scale = 0.009648532800137615
        #self.output = output
    
    @staticmethod
    def atomic_charges(P, n_orbital=4):
        """
        get atomic charge based on single-particle density matrix P
        n_orbital : number of orbitals for each atom, default is 4
        """
        n_molecule = P.shape[0]
        n_atom = P.shape[1]//n_orbital
        q = P.diagonal(dim1=1,dim2=2).reshape(n_molecule, n_atom, n_orbital).sum(dim=2)
        return q
    
    @staticmethod
    def dipole(q, coordinates, com2zero=True):
        R = coordinates - coordinates.mean(dim=1).unsqueeze(1) if com2zero else coordinates
        return torch.sum( q.unsqueeze(2) * R, dim=1 )
    
    def forward(self, molecule, learned_parameters=dict(), xl_bomd_params=dict(), P0=None, err_threshold = None, max_rank = None, T_el = None, dm_prop='SCF', *args, **kwargs):
        """
        return force in unit of eV/Angstrom
        return force, density matrix, total energy of this batch
        """
        if dm_prop=='SCF':
            molecule.force, P, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, C, self.charge, self.notconverged = \
                        self.conservative_force(molecule, P0=P0, learned_parameters=learned_parameters, *args, **kwargs)
            if self.do_hessian:
                # vmap doesn't work due to dynamic shapes in PYSEQM
                # as a result, this is quite inefficient for batches!
                _3n = molecule.molsize * 3
                e_vecs = torch.eye(_3n, device=molecule.force.device)
                g_flat = -molecule.force.view(molecule.nmol, _3n)
                molecule.hessian = torch.stack([torch.stack(
                                                   [adgrad(g, molecule.coordinates, ev, retain_graph=True)[0][i]
                                                for ev in e_vecs]) for i, g in enumerate(g_flat)]
                                              ).view(molecule.nmol, _3n, _3n)
                
                if not self.keep_graph: molecule.force.detach_()
                curve, molecule.vib_modes = harmonic_analysis(molecule.mass, molecule.coordinates,
                                                              molecule.hessian)
                molecule.vib_freqs = nestedmap_unary(curve2freq, curve)
            molecule.dm = P.detach()
            molecule.mo_coeff = C#.detach() # detach needed?
            
        elif dm_prop=='XL-BOMD':
            molecule.force, molecule.dm, molecule.Hf, molecule.Etot, molecule.Eelec, molecule.Enuc, molecule.Eiso, molecule.e_mo, molecule.e_gap, \
            molecule.Electronic_entropy, molecule.dP2dt2, molecule.Krylov_Error,  molecule.Fermi_occ = \
                        self.conservative_force_xl(molecule, P0, learned_parameters, xl_bomd_params=xl_bomd_params, *args, **kwargs)

        with torch.no_grad():
            # $$$
            if molecule.dm.dim() == 4:
                molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm[:,0])
                molecule.q -= self.atomic_charges(molecule.dm[:,1]) # unit +e, i.e. electron: -1.0
            else:
                molecule.q = molecule.const.tore[molecule.species] - self.atomic_charges(molecule.dm) # unit +e, i.e. electron: -1.0
            molecule.d = self.dipole(molecule.q, molecule.coordinates)
            



        #return F, P, L

    def get_force(self):
        return self.force

    def get_dm(self):
        return self.P

    def get_Hf(self):
        return self.Hf
    
    def get_Electronic_entropy(self):
        return self.El_Ent
    
    def get_dP2dt2(self):
        return self.dP2dt2

    def get_Krylov_Error(self):
        return self.Error

    def get_e_gap(self):
        return self.e_gap
    
    def get_e_mo(self):
        return self.e_mo

    def get_Fermi_occ(self):
        return self.Fermi_occ

    # def get_charger(self, xx):

    #     return charge

