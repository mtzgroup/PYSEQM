import torch
from warnings import warn


class _def_get_mu_base(torch.autograd.Function):
    """
    Base class for obtaining chemical potential (mu) with
    custom backward according to implicit differentiation.
    """
    @staticmethod
    def smearing(eps, mu, kT=torch.tensor(0.05)):
        raise NotImplementedError
    
    @staticmethod
    def dn_dmu(eps, mu, kT, f):
        raise NotImplementedError
    
    @staticmethod
    def dn_de(eps, mu, kT, f):
        raise NotImplementedError
    
    @staticmethod
    def dn_dkt(eps, mu, kT):
        raise NotImplementedError
    
    @classmethod
    def forward(cls, ctx, eps, kT, n_el, tol=1e-10, maxiter=100):
        """
        Solve for chemical potential to match total number of electrons
        
        Parameters
        ----------
        eps: torch.Tensor
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
        f_smear = cls.smearing
        mu_lo = eps.min(dim=-1).values - 50 * kT
        mu_hi = eps.max(dim=-1).values + 50 * kT
        mu = 0.5 * (mu_lo + mu_hi)
        converged = False
        ## try Newton
        for it in range(maxiter):
            f = f_smear(eps, mu, kT=kT)
            delta = f.sum(dim=-1) - n_el
            dN_dmu = cls.dn_dmu(eps, mu, kT, f)
            ## for low kT, Newton can fail and overshoot -> do bisection
            if (dN_dmu.abs() < 1e-4).any(): break
            mu_new = mu - delta / dN_dmu
            if torch.all(torch.abs(mu_new - mu) < tol):
                converged = True
                break
            mu = mu_new
        
        # fallback to bisection
        if not converged:
            # make sure root is in bracket
            for it in range(10):
                N_lo = f_smear(eps, mu_lo, kT=kT).sum(dim=-1)
                c_lo = N_lo + 1e-6 > n_el
                N_hi = f_smear(eps, mu_hi, kT=kT).sum(dim=-1)
                c_hi = N_hi - 1e-6 < n_el
                if not ( c_lo.any() or c_hi.any() ): break
                mu_lo = torch.where(c_lo, mu_lo - 100 * kT, mu_lo)
                mu_hi = torch.where(c_hi, mu_hi + 100 * kT, mu_hi)
            if ( c_lo.any() or c_hi.any() ):
                raise RuntimeError("Failed to bracket root in `get_chemical_potential`.")
            
            mu = 0.5 * (mu_lo + mu_hi)
            for it in range(maxiter):
                N_mid = f_smear(eps, mu, kT=kT).sum(dim=-1)
                mu_lo = torch.where(N_mid > n_el, mu_lo, mu)
                mu_hi = torch.where(N_mid < n_el, mu_hi, mu)
                mu = 0.5 * (mu_lo + mu_hi)
                if ((mu_hi - mu_lo) < tol).all():
                    mu_new, converged = mu, True
                    break
        if not converged:
            warn("Could not converge chemical potential to desired threshold. Returning best guess.")
            mu_new = mu
        ctx.save_for_backward(mu_new, eps, kT)
        return mu_new

    @classmethod
    def backward(cls, ctx, mu_bar):
        mu_sol, eps, kT = ctx.saved_tensors
        f = cls.smearing(eps, mu_sol, kT=kT)
        dN_deps = cls.dn_de(eps, mu_sol, kT, f)
        mu_unsq = mu_sol.unsqueeze(-1)
        dN_dkT = cls.dn_dkt(eps, mu_unsq, kT)
        dN_dmu = cls.dn_dmu(eps, mu_sol, kT, f).unsqueeze(-1)
        dmu_de = -dN_dmu.pow(-1) * dN_deps
        dmu_dkT = -dN_dmu.pow(-1) * dN_dkT
        mu_bar_unsq = mu_bar.unsqueeze(-1)
        return mu_bar_unsq * dmu_de, mu_bar_unsq * dmu_dkT, None, None, None
    

