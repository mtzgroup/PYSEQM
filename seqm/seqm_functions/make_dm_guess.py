import torch
from warnings import warn
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from seqm.basics import Pack_Parameters
from .pack import *
from .diag import DEGEN_EIGENSOLVER, degen_symeig, pytorch_symeig#sym_eig_trunc, sym_eig_trunc1, pseudo_diag


CHECK_DEGENERACY = False


def diag_guess(molecule, seqm_parameters, learned_parameters=dict(),
               overwrite_existing_dm=False, **kwargs):
    dtype  = molecule.xij.dtype
    device = molecule.xij.device
    nH = torch.count_nonzero(molecule.species == 1)
    orb_species = molecule.species.repeat_interleave(4,1).flatten()
    diags = torch.zeros_like(orb_species, dtype=dtype, device=device)
    diags[orb_species > 1] = molecule.const.tore[orb_species[orb_species > 1]] / 4.0
    diags[orb_species == 1] = torch.tensor([1.,0.,0.,0.], dtype=dtype, device=device).repeat(nH)
    diags_mol = diags.reshape(molecule.nmol, -1)
    if molecule.nocc.dim() == 2:
        diags_mol = torch.stack((0.5 * diags_mol, 0.5 * diags_mol), dim=1)
        P_blank = torch.zeros((molecule.nmol, 2, 4*molecule.molsize, 4*molecule.molsize),
                              dtype=dtype, device=device)
    else:
        P_blank = torch.zeros((molecule.nmol, 4*molecule.molsize, 4*molecule.molsize),
                              dtype=dtype, device=device)
    return P_blank.diagonal_scatter(diags_mol, dim1=-2, dim2=-1)


def make_dm_guess(molecule, seqm_parameters, mix_homo_lumo=False, mix_coeff=0.4,
                  learned_parameters=dict(), overwrite_existing_dm=False,
                  from_hcore=True, ivans_beta=False):
    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    dtype  = molecule.xij.dtype
    device = molecule.xij.device
    
    if not torch.is_tensor(molecule.dm) or overwrite_existing_dm:
        P = diag_guess(molecule, seqm_parameters, learned_parameters=learned_parameters,
                       overwrite_existing_dm=overwrite_existing_dm)
    else:
        P = molecule.dm
    
    if not from_hcore:
        molecule.dm = P
        return P, None
    
    packpar = Pack_Parameters(seqm_parameters).to(device)
    
    if callable(learned_parameters):
        adict = learned_parameters(molecule.species, molecule.coordinates)
        parameters = packpar(molecule.Z, learned_params=adict)
    else:
        parameters = packpar(molecule.Z, learned_params=learned_parameters)

    beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)), dim=1)
    Kbeta = parameters.get('Kbeta', None)
    zetas = parameters['zeta_s']
    zetap = parameters['zeta_p']
    uss = parameters['U_ss']
    upp = parameters['U_pp']
    gss = parameters['g_ss']
    gsp = parameters['g_sp']
    gpp = parameters['g_pp']
    gp2 = parameters['g_p2']
    hsp = parameters['h_sp']
    
    M, w = hcore(molecule.const, molecule.nmol, molecule.molsize, molecule.maskd, molecule.mask,
                 molecule.idxi, molecule.idxj, molecule.ni, molecule.nj, molecule.xij,
                 molecule.rij, molecule.Z, zetas, zetap, uss, upp, gss, gpp, gp2, hsp,
                 beta, molecule.nHeavy, molecule.nHydro, molecule.nocc, Kbeta=Kbeta,
                 ivans_beta=ivans_beta)
    
    if molecule.nocc.dim() == 2:
        x = fock_u_batch(molecule.nmol, molecule.molsize, P, M, molecule.maskd, molecule.mask,
                         molecule.idxi, molecule.idxj, w, gss, gpp, gsp, gp2, hsp)
        
        nheavyatom = molecule.nHeavy.repeat_interleave(2)
        nH = molecule.nHydro.repeat_interleave(2)
        nocc = molecule.nocc.flatten()
        #Gershgorin circle theorem estimate upper bounds of eigenvalues  
        x_orig_shape = x.size()
        x0 = pack(x, nheavyatom, nH)
        nmol, size, _ = x0.shape
        
        aii = x0.diagonal(dim1=1, dim2=2)
        ri = torch.sum(torch.abs(x0), dim=2) - torch.abs(aii)
        hN = torch.max(aii + ri, dim=1)[0]
        dE = hN - torch.min(aii - ri, dim=1)[0] #(maximal - minimal) get range
        norb = nheavyatom * 4 + nH
        pnorb = size - norb
        nn = torch.max(pnorb).item()
        dx = 0.005
        mutipler = torch.arange(1.0+dx, 1.0+nn*dx+dx, dx, dtype=dtype, device=device)[:nn]
        ind = torch.arange(size, dtype=torch.int64, device=device)
        cond = pnorb>0
        for i in range(nmol):
            if cond[i]:
                x0[i,ind[norb[i]:], ind[norb[i]:]] = mutipler[:pnorb[i]]*dE[i]+hN[i]
        try:
            e0, v = sym_eigh(x0)
        except:
            if torch.isnan(x0).any(): print(x0)
            e0, v = sym_eigh(x0)
        
        if mix_homo_lumo:
            mix_coeff = torch.tensor([mix_coeff], device=device)
            vv = v.view(int(v.shape[0]/2), 2, v.shape[1], v.shape[2])
            ## alpha channel
            _lumo_idx = molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).transpose(0, -1)
            lumo_idx = _lumo_idx.repeat(1, vv.shape[-1], 1)
            homo_idx = (lumo_idx - 1).clamp_min(0)
            v_lumo = vv[:,0].gather(2, lumo_idx)
            v_homo = vv[:,0].gather(2, homo_idx)
            v_mix = (1 - mix_coeff) * v_homo + mix_coeff * v_lumo
            vv[:,0].scatter_(2, homo_idx, v_mix)
            ## beta channel
            _lumo_idx = molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).transpose(0, -1)
            lumo_idx = _lumo_idx.repeat(1, vv.shape[-1], 1)
            homo_idx = (lumo_idx - 1).clamp_min(0)
            v_lumo = vv[:,1].gather(2, lumo_idx)
            v_homo = vv[:,1].gather(2, homo_idx)
            v_mix = (1 - mix_coeff) * v_homo + mix_coeff * v_lumo
            vv[:,1].scatter_(2, homo_idx, v_mix)
            
            v = vv.view(int(vv.shape[0]*2), vv.shape[2], vv.shape[3])
            
        if CHECK_DEGENERACY:
            e = torch.zeros((nmol, x.shape[-1]), dtype=dtype, device=device)
            e[...,:size] = e0
            for i in range(nmol):
                if cond[i]: e[i,norb[i]:size] = 0.0

            e = e.view(x_orig_shape[0:3])
            t = torch.stack(list(map(lambda a,b,n : construct_P(a, b, n), e, v, nocc)))
        else:
            t = 2.0*torch.stack(list(map(lambda a,n : torch.matmul(a[:,:n], a[:,:n].transpose(0,1)), v, nocc)))

        P = unpack(t, nheavyatom, nH, x.shape[-1])
        v = v.view(int(v.shape[0]/2), 2, v.shape[1], v.shape[2])
        P = P.view(x_orig_shape) / 2
        molecule.dm = P
        return P, v
    else:
        x = fock(molecule.nmol, molecule.molsize, P, M, molecule.maskd, molecule.mask, molecule.idxi,
                 molecule.idxj, w, gss, gpp, gsp, gp2, hsp)

        x0 = pack(x, molecule.nHeavy, molecule.nHydro)
        nmol, size, _ = x0.shape

        aii = x0.diagonal(dim1=1, dim2=2)
        abs_x0 = torch.abs(x0)
        abs_aii = torch.abs(aii)
        ri = torch.sum(abs_x0, dim=2) - abs_aii
        hN = torch.max(aii + ri, dim=1)[0]
        dE = hN - torch.min(aii - ri, dim=1)[0]
        norb = molecule.nHeavy * 4 + molecule.nHydro
        pnorb = size - norb
        nn = torch.max(pnorb).item()
        dx = 0.005
        mutipler = torch.arange(1.0+dx, 1.0+nn*dx+dx, dx, dtype=dtype, device=device)[:nn]
        ind = torch.arange(size, dtype=torch.int64, device=device)
        cond = pnorb>0
        for i in range(nmol):
            if cond[i]:
                x0[i,ind[norb[i]:], ind[norb[i]:]] = mutipler[:pnorb[i]] * dE[i] + hN[i]
        try:
            e0, v = sym_eigh(x0)
        except:
            if torch.isnan(x0).any():
#                raise torch._C._LinAlgError("Input to eigh contains NaN")
                warn("Input to eigh contains NaN. Returning diagonal guess")
                molecule.dm = P
                return P, None
            else:
                e0, v = sym_eigh(x0)
        
        if mix_homo_lumo:
            lumo_idx = molecule.nocc.unsqueeze(0).unsqueeze(0).transpose(0,-1)
            gather_idx = lumo_idx.repeat(1,v.shape[-1],1)
            v_lumo = v.gather(2, gather_idx)
            homo_idx = gather_idx - 1
            v_homo = v.gather(2, homo_idx)
            mix_coeff = torch.tensor([mix_coeff], device=device)
            v_mix = (1 - mix_coeff) * v_homo + mix_coeff * v_lumo
            v.scatter_(2, homo_idx, v_mix)
            
        if CHECK_DEGENERACY:
            e = torch.zeros((nmol, x.shape[-1]), dtype=dtype, device=device)
            e[...,:size] = e0
            for i in range(nmol):
                if cond[i]: e[i,norb[i]:size] = 0.0
            t = torch.stack(list(map(lambda a,b,n : construct_P(a, b, n), e, v, molecule.nocc)))
        else:
            t = 2.0*torch.stack(list(map(lambda a,n : torch.matmul(a[:,:n], a[:,:n].transpose(0,1)), v, molecule.nocc)))

        P = unpack(t, molecule.nHeavy, molecule.nHydro, x.shape[-1])
        molecule.dm = P
        return P, v
    
