import torch


########################
#  integer occupations #
########################

def integer_occ(e, packed_shape, nocc):
    orb_idx = torch.arange(packed_shape[-1], dtype=e.dtype, device=e.device).repeat(*packed_shape[:-2],1)
    is_occ = orb_idx < nocc.unsqueeze(-1)
    f = torch.zeros(*packed_shape[:-1], dtype=e.dtype, device=e.device)
    f[is_occ] = 2.
    return f


############################################
#  split over degenerate HOMOs (ROHF-like) #
############################################

def dm_smeared_degen(e, v, nocc):
    atol = 1.0e-7 if e.dtype==torch.float32 else 1.0e-14
    cond = (e - e[nocc-1]).abs() < atol
    if cond[nocc:].any():
        c = torch.nonzero(cond)
        indx1 = c[0].item()
        indx2 = c[-1].item()+1
        nd = indx2 - indx1
        coeff = torch.ones(1, indx2, device=e.device, dtype=e.dtype)
        coeff[0,indx1:] = (nocc.type(torch.double) - indx1) / nd
        return 2.0 * torch.matmul(coeff * v[:,:indx2], v[:,:indx2].transpose(0,1))
    else:
        return 2.0 * torch.matmul(v[:,:nocc], v[:,:nocc].transpose(0,1))

def smeared_degen_occ(e, packed_shape, nocc):
    raise NotImplementedError


################
#  FON / FOMO  #
################

def fermi_dirac(e, mu, kT=0.05, clamp_at=54.5):
    x = (mu.unsqueeze(-1) - e) / kT
    return torch.sigmoid( x.clamp(-clamp_at, clamp_at) )

def get_chemical_potential(e, n_el, kT=0.05, tol=1e-9, maxiter=100, clamp_fermi=54.5):
    """ implicit backward? """
    n_mol, norb = e.shape
    mu_lo = e.min(dim=1).values - 50 * kT
    mu_hi = e.max(dim=1).values + 50 * kT
    mu = 0.5 * (mu_lo + mu_hi)
    mu_old = mu.clone()
    notconv = torch.ones(n_mol, dtype=bool, device=e.device)
    for i in range(maxiter):
        f = fermi_dirac(e[notconv], mu[notconv], kT=kT)
        N_sum = 2. * f.sum(dim=1)
        dN_dmu = -2. / kT * (f * (1 - f)).sum(1)
        dmu = (n_el[notconv] - N_sum) / dN_dmu.clamp_min(1e-16)
        mu_new = mu[notconv] + dmu

        oob = (mu_new < mu_lo[notconv]) | (mu_new > mu_hi[notconv])
        mu_mid = 0.5 * (mu_lo[notconv] + mu_hi[notconv])
        mu[notconv] = torch.where(oob, mu_mid, mu_new)
        too_hi = N_sum > n_el[notconv]
        mu_lo[notconv] = torch.where(too_hi,  mu[notconv], mu_lo[notconv])
        mu_hi[notconv] = torch.where(~too_hi, mu[notconv], mu_hi[notconv])

        notconv = (mu - mu_old).abs() > tol
        if not notconv.any(): break
        mu_old = mu.clone()
    return mu

def fractional_occ(e, packed_shape, nocc, kT=0.05):
    if e.dim() > 2:
        raise NotImplementedError("Fractional occupations does not support UHF yet")
    n_el = 2. * nocc
    e_real = e[...,:packed_shape[-1]]
    mu = get_chemical_potential(e_real, n_el, kT=kT)
    return fermi_dirac(e_real, mu, kT=kT)

