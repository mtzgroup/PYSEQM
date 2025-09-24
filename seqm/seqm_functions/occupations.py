import torch
from warnings import warn


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


class get_chemical_potential(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, n_el, kT, tol=1e-9, maxiter=100):
        """
        Solve for chemical potential in Fermi-Dirac occupations
        
        Parameters
        ----------
        e: torch.Tensor
            orbital energies
        n_el: torch.Tensor
            number of occupied orbitals
        kT: float
            smearing width (KB * electronic temperature)
        tol: float
            convergence threshold
        maxiter: int
            maximum number of iterations
        """
        mu_lo = e.min(dim=-1).values - 50 * kT
        mu_hi = e.max(dim=-1).values + 50 * kT
        mu = 0.5 * (mu_lo + mu_hi)
        converged = False
        for it in range(maxiter):
            f = fermi_dirac(e, mu, kT=kT)
            delta = f.sum(dim=-1) - n_el
            dN_dmu = kT.pow(-1) * torch.sum(f * (1 - f), dim=-1)
            ## for low kT, Newton might fail and overshoot -> resort to bisection
            if (dN_dmu.abs() < 1e-4).any(): break
            mu_new = mu - delta / dN_dmu
            if torch.all(torch.abs(mu_new - mu) < tol):
                converged = True
                break
            mu = mu_new
        
        if not converged:
            # make sure root is in bracket
            for it in range(10):
                N_lo = fermi_dirac(e, mu_lo, kT=kT).sum(dim=-1)
                c_lo = N_lo + 1e-6 > n_el
                N_hi = fermi_dirac(e, mu_hi, kT=kT).sum(dim=-1)
                c_hi = N_hi - 1e-6 < n_el
                if not ( c_lo.any() or c_hi.any() ): break
                mu_lo = torch.where(c_lo, mu_lo - 100 * kT, mu_lo)
                mu_hi = torch.where(c_hi, mu_hi + 100 * kT, mu_hi)
            if ( c_lo.any() or c_hi.any() ):
                raise RuntimeError("Failed to bracket root in `get_chemical_potential`.")
            
            # fallback to bisection
            mu = 0.5 * (mu_lo + mu_hi)
            for it in range(maxiter):
                N_mid = fermi_dirac(e, mu, kT=kT).sum(dim=-1)
                mu_lo = torch.where(N_mid > n_el, mu_lo, mu)
                mu_hi = torch.where(N_mid < n_el, mu_hi, mu)
                mu = 0.5 * (mu_lo + mu_hi)
                if ((mu_hi - mu_lo) < 2. * tol).all():
                    mu_new, converged = mu, True
                    break
        if not converged:
            warn("Could not converge chemical potential to desired threshold. Returning best guess.")
            mu_new = mu
        ctx.save_for_backward(mu_new, e, kT)
        return mu_new

    @staticmethod
    def backward(ctx, mu_bar):
        mu_sol, e, kT = ctx.saved_tensors
        f = fermi_dirac(e, mu_sol, kT=kT)
        dN_de = kT.pow(-1) * f * (f - 1)
        mu_unsq = mu_sol.unsqueeze(-1)
        nom = (e - mu_unsq) * torch.exp( (e + mu_unsq)/kT )
        denom = kT * ( torch.exp(e/kT) + torch.exp(mu_unsq/kT) )
        dN_dkT = (nom / denom.pow(2)).sum(dim=-1)
        dN_dmu = kT.pow(-1) * torch.sum(f * (1 - f), dim=-1).unsqueeze(-1)
        dmu_de = -dN_dmu.pow(-1) * dN_de
        dmu_dkT = -dN_dmu.pow(-1) * dN_dkT
        mu_bar_unsq = mu_bar.unsqueeze(-1)
        return mu_bar_unsq * dmu_de, None, mu_bar_unsq * dmu_dkT, None, None

    

def fractional_occ(e, packed_shape, n_occ, kT=0.05):
    e_real = e[...,:packed_shape[-1]]
    mu = get_chemical_potential().apply(e_real, n_occ, kT)
    return 2. * fermi_dirac(e_real, mu, kT=kT)

