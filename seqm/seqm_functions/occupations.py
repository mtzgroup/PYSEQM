import torch
from math import sqrt
from .chemical_potential import _def_get_mu_base


########################
#  integer occupations #
########################

def integer_occ(eps, packed_shape, nocc):
    orb_idx = torch.arange(packed_shape[-1], dtype=eps.dtype, device=eps.device).repeat(*packed_shape[:-2],1)
    is_occ = orb_idx < nocc.unsqueeze(-1)
    f = torch.zeros(*packed_shape[:-1], dtype=eps.dtype, device=eps.device)
    f[is_occ] = 2.
    return f


############################################
#  split over degenerate HOMOs (ROHF-like) #
############################################

def dm_smeared_degen(eps, v, nocc):
    atol = 1.0e-7 if eps.dtype==torch.float32 else 1.0e-14
    cond = (eps - eps[nocc-1]).abs() < atol
    if cond[nocc:].any():
        c = torch.nonzero(cond)
        indx1 = c[0].item()
        indx2 = c[-1].item()+1
        nd = indx2 - indx1
        coeff = torch.ones(1, indx2, device=eps.device, dtype=eps.dtype)
        coeff[0,indx1:] = (nocc.type(torch.double) - indx1) / nd
        return 2.0 * torch.matmul(coeff * v[:,:indx2], v[:,:indx2].transpose(0,1))
    else:
        return 2.0 * torch.matmul(v[:,:nocc], v[:,:nocc].transpose(0,1))

def smeared_degen_occ(eps, packed_shape, nocc):
    raise NotImplementedError


################
#  FON / FOMO  #
################

# Fermi-Dirac occupations
def fermi_dirac(eps, mu, kT=torch.tensor(0.05)):
    x = (mu.unsqueeze(-1) - eps) * kT.pow(-1)
    return torch.sigmoid(x)

class _def_get_mu_fermi(_def_get_mu_base):
    @staticmethod
    def smearing(eps, mu, kT=torch.tensor(0.05)):
        x = (mu.unsqueeze(-1) - eps) * kT.pow(-1)
        return torch.sigmoid( x )
    
    @staticmethod
    def dn_dmu(eps, mu, kT, f):
        return (kT.pow(-1) * f * (1 - f)).sum(dim=-1)
    
    @staticmethod
    def dn_de(eps, mu, kT, f):
        return kT.pow(-1) * f * (f - 1)
    
    @staticmethod
    def dn_dkt(eps, mu, kT):
        kT_inv = kT.pow(-1)
        nom = (eps - mu) * torch.exp( (eps + mu) * kT_inv )
        denom = kT * ( torch.exp(eps * kT_inv) + torch.exp(mu * kT_inv) )
        return (nom / denom.pow(2)).sum(dim=-1)

    
# Occupations from "level broadening" [see doi.org/10.1063/1.3436501]
def erfc_occ(eps, mu, kT=torch.tensor(0.05)):
    x = (eps - mu) / sqrt(2.) * kT.pow(-1)
    return torch.erfc(x) / 2

class _def_get_mu_erfc(_def_get_mu_base):
    @staticmethod
    def smearing(eps, mu, kT=torch.tensor(0.05)):
        x = (eps - mu) / sqrt(2.) * kT.pow(-1)
        return torch.erfc(x) / 2
    
    @staticmethod
    def dn_dmu(eps, mu, kT, f):
        x = -((eps - mu) * kT.pow(-1)).pow(2) / 2
        terms = torch.exp(x) / sqrt(2. * torch.pi) * kT.pow(-1)
        return torch.sum(terms, dim=-1)
    
    @staticmethod
    def dn_de(eps, mu, kT, f):
        x = -((eps - mu) * kT.pow(-1)).pow(2) / 2
        return -torch.exp(x) / sqrt(2. * torch.pi) * kT.pow(-1)
    
    @staticmethod
    def dn_dkt(eps, mu, kT):
        x = -((eps - mu) * kT.pow(-1)).pow(2) / 2
        terms = torch.exp(x) / sqrt(2. * torch.pi) * kT.pow(-2)
        return torch.sum((eps - mu) * terms, dim=-1)
    

def fractional_occ(eps, packed_shape, n_occ, kT=torch.tensor(0.05),
                   smearing="fermi"):
    if smearing.lower() in ["fermi", "fermi_dirac"]:
        f_smear = fermi_dirac
        get_chemical_potential = _def_get_mu_fermi.apply
    elif smearing.lower() in ["erf", "erfc"]:
        f_smear = erfc_occ
        get_chemical_potential = _def_get_mu_erfc.apply
    else:
        raise ValueError("Unknown smearing '"+smearing+"'.")
    e_real = eps[...,:packed_shape[-1]]
    mu = get_chemical_potential(e_real, kT, n_occ)
    return 2. * f_smear(e_real, mu, kT=kT)
    

