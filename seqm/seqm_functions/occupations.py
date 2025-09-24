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
    return torch.sigmoid( x )


def get_chemical_potential(e, n_el, kT=0.05, tol=1e-9, maxiter=100, clamp_fermi=54.5):
    """ (implicit) backward? """
    n_mol, norb = e.shape
    mu_lo = e.min(dim=1).values - 50 * kT
    mu_hi = e.max(dim=1).values + 50 * kT
    mu = 0.5 * (mu_lo + mu_hi)
    for it in range(maxiter):
        f = fermi_dirac(e, mu, kT=kT)
        delta = 2.0 * f.sum(dim=1) - n_el
        dN_dmu = 2.0 * torch.sum(f * (1 - f), dim=1) / kT
        ## for low kT, Newton might fail and overshoot -> resort to bisection
        if (dN_dmu.abs() < 1e-4).any():
            break
        mu_new = mu - delta / dN_dmu
        if torch.all(torch.abs(mu_new - mu) < tol):
            return mu_new
        mu = mu_new
    
    # make sure root is in bracket
    for it in range(10):
        N_lo = 2.0 * fermi_dirac(e, mu_lo, kT=kT).sum(dim=1)
        c_lo = N_lo + 1e-6 > n_el
        N_hi = 2.0 * fermi_dirac(e, mu_hi, kT=kT).sum(dim=1)
        c_hi = N_hi - 1e-6 < n_el
        if not ( c_lo.any() or c_hi.any() ): break
        mu_lo = torch.where(c_lo, mu_lo - 100 * kT, mu_lo)
        mu_hi = torch.where(c_hi, mu_hi + 100 * kT, mu_hi)
    if ( c_lo.any() or c_hi.any() ):
        raise RuntimeError("Failed to bracket root in `get_chemical_potential`.")
    
    # fallback to bisection
    mu = 0.5 * (mu_lo + mu_hi)
    for it in range(maxiter):
        N_mid = 2.0 * fermi_dirac(e, mu, kT=kT).sum(dim=1)
        mu_lo = torch.where(N_mid > n_el, mu_lo, mu)
        mu_hi = torch.where(N_mid < n_el, mu_hi, mu)
        mu = (mu_lo + mu_hi) / 2
        if (mu_hi - mu_lo) / 2 < tol:
            return mu

    return mu

def fractional_occ(e, packed_shape, nocc, kT=0.05):
    if e.dim() > 2:
        raise NotImplementedError("Fractional occupations does not support UHF yet")
    n_el = 2. * nocc
    e_real = e[...,:packed_shape[-1]]
    mu = get_chemical_potential(e_real, n_el, kT=kT)
    f = 2. * fermi_dirac(e_real, mu, kT=kT)
    return f

