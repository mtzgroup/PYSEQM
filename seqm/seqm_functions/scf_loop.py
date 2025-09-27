import torch
from torch.autograd import grad as agrad
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .pack import *
from .diag import sym_eig_trunc, sym_eig_trunc1
from .occupations import integer_occ, smeared_degen_occ, fractional_occ
from .diis import SimpleDIIS
from .orig_scf_forwards import scf_forward1, scf_forward2, scf_forward3
import warnings
import time

#scf_backward==0: ignore the gradient on density matrix
#scf_backward==1: use recursive formula/implicit autodiff
#scf_backward==2: go backward scf loop directly


debug = False
RAISE_ERROR_IF_SCF_FORWARD_FAILS = False
RAISE_ERROR_IF_SCF_BACKWARD_FAILS = False

# tolerance, max no. iteration, and history size for Anderson acceleration
# in solving fixed point problem
SCF_BACKWARD_ANDERSON_TOLERANCE = 1e-4  # this seems stable enough, but TODO!
SCF_BACKWARD_ANDERSON_MAXITER = 50      # sufficient for all test cases
SCF_BACKWARD_ANDERSON_HISTSIZE = 5      # seems reasonable, but TODO!



def build_dm(e, v, nocc, occ_mode=0, occ_kT=torch.tensor(0.05), smearing="fermi"):
    """
    occ_mode: int
        occupation type, 0: integer, 1: split among degenerate HOMOs, 2: FON/FOMO/Fermi
    """
    packed_shape = v.shape
    if occ_mode == 0:
        f = integer_occ(e, packed_shape, nocc)
    elif occ_mode == 1:
        f = smeared_degen_occ(e, packed_shape, nocc)
    elif occ_mode == 2:
        f = fractional_occ(e, packed_shape, nocc, kT=occ_kT, smearing=smearing)
    else:
        raise ValueError("Invalid `occ_mode`. Expected one of 0, 1, or 2, but got "+str(occ_mode)+".")
    D = torch.einsum("...ij,...j,...kj->...ik", v, f, v)
    return D


# TODO: abstract and modularize SCF methods (only Fock update differs)!

# simple DIIS
def scf_diis(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
             nmol, molsize, maskd, mask, idxi, idxj, P, eps_E,
             sp2=[False], alpha=0.0, backward=False, scf_maxiter=200,
             occ_mode="integer", occ_kT=torch.tensor(0.05), smearing="fermi", 
             eps_P=1e-5, diis_start=2, diis_max=8, detach_diis=True,
             compress_rank=None):
    notconv = torch.ones(nmol, dtype=torch.bool, device=M.device)
    if P.dim() == 4:
        get_fock_mat, n_spin = fock_u_batch, 2
    else:
        get_fock_mat, n_spin = fock, 1
    
    ## DIIS needs at least two warm-up steps
    diis_start = max(diis_start, 2)
    delta_E = torch.full((nmol,), 1e3, dtype=P.dtype, device=P.device)
    delta_P = torch.full((nmol,n_spin), 1e3, dtype=P.dtype, device=P.device)
    F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss,
                     gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4).transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eel = elec_energy(P, F, Hcore)
    Eel_old = Eel.clone()
    
    extrapolator = SimpleDIIS(n_mol=nmol, nspin=n_spin, n=4*molsize, k_max=diis_max,
                        compress_rank=compress_rank,
                        device=M.device, dtype=M.dtype)
    Pold = P.clone()
    if debug: print("Iter  delta E [eV]    delta P    unconv dt [s]")
    for k in range(1, scf_maxiter + 1):
        if debug: start_time = time.time()
        e, v = sym_eig_trunc(F[notconv], nHeavy[notconv], nHydro[notconv],
                             nOccMO[notconv])
        D = build_dm(e, v, nOccMO[notconv], occ_mode=occ_mode, occ_kT=occ_kT, smearing=smearing) / n_spin
        P[notconv] = unpack(D, nHeavy[notconv], nHydro[notconv], F.shape[-1])
        ### Fock update
        F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss,
                         gpp, gsp, gp2, hsp)
        
        FP = torch.einsum("...ij,...jk->...ik", F, P) 
        PF = torch.einsum("...ij,...jk->...ik", P, F)
        FP_commutator = FP - PF
        extrapolator.append(F, FP_commutator)
        if k >= diis_start:
            F_diis = extrapolator.extrapolate()
            if detach_diis:
                # trick to replace data in F with the data in F_diis,
                # while keeping the grad field of F
                dF = F - F_diis.detach()
                F = F_diis.detach() + dF
            else:
                F = F_diis
        ###############
        
        delta_P[notconv] = (P.abs() - Pold.abs()).norm(dim=(-2,-1))
        deltaP_mol = (delta_P > eps_P).any(dim=-1) # over spin channels
        Pold[notconv] = P[notconv]
        
        Eel[notconv] = elec_energy(P[notconv], F[notconv], Hcore[notconv])
        delta_E[notconv] = torch.abs(Eel[notconv] - Eel_old[notconv])
        Eel_old[notconv] = Eel[notconv]
        
        notconv = (delta_E > eps_E).logical_or(deltaP_mol)
        if debug:
            end_time = time.time()
            print(" {:3d}  {:8.5e}   {:8.5e}  {:5d}  {:5.3f}".format(k, delta_E.max().item(),
                    delta_P.max().item(), notconv.sum().item(), end_time - start_time) )
        if not notconv.any(): break
    
    ## if returning F, probably need to re-built Fock!!
    return P, notconv
    


# constant mixing
def scf_constmix(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                 nmol, molsize, maskd, mask, idxi, idxj, P, eps_E,
                 sp2=[False], alpha=0.0, backward=False, scf_maxiter=200,
                 occ_mode=0, occ_kT=torch.tensor(0.05), smearing="fermi",
                 eps_P=1e-5):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    notconv = torch.ones(nmol, dtype=torch.bool, device=M.device)
    if P.dim() == 4:
        get_fock_mat, n_spin = fock_u_batch, 2
    else:
        get_fock_mat, n_spin = fock, 1
    
    delta_E = torch.ones(nmol, dtype=P.dtype, device=P.device)
    delta_P = torch.full((nmol,n_spin), 1e3, dtype=P.dtype, device=P.device)
    F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4).transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eel = elec_energy(P, F, Hcore)
    Eel_old = Eel.clone()
    Pold = P.clone()
    if debug: print("Iter  delta E [eV]    delta P    unconv dt [s]")
    for k in range(1, scf_maxiter + 1):
        if debug: start_time = time.time()
        e, v = sym_eig_trunc(F[notconv], nHeavy[notconv], nHydro[notconv],
                             nOccMO[notconv])
        D = build_dm(e, v, nOccMO[notconv], occ_mode=occ_mode, occ_kT=occ_kT, smearing=smearing) / n_spin
        P[notconv] = unpack(D, nHeavy[notconv], nHydro[notconv], F.shape[-1])
        ### Fock update
        if backward:
            P = alpha * Pold + (1.0 - alpha) * P
        else:
            P[notconv] = alpha * Pold[notconv] + (1.0 - alpha) * P[notconv]
        
        F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, 
                         gpp, gsp, gp2, hsp)
        ###############

        delta_P[notconv] = (P.abs() - Pold.abs()).norm(dim=(-2,-1))
        deltaP_mol = (delta_P > eps_P).any(dim=-1) # over spin channels
        Pold[notconv] = P[notconv]

        Eel[notconv] = elec_energy(P[notconv], F[notconv], Hcore[notconv])
        delta_E[notconv] = torch.abs(Eel[notconv] - Eel_old[notconv])
        Eel_old[notconv] = Eel[notconv]

        notconv = (delta_E > eps_E).logical_or(deltaP_mol)
        if debug:
            end_time = time.time()
            print(" {:3d}  {:8.5e}   {:8.5e}  {:5d}  {:5.3f}".format(k, delta_E.max().item(),
                    delta_P.max().item(), notconv.sum().item(), end_time - start_time) )
        if not notconv.any(): break
    return P, notconv


        


def fixed_point_anderson(fp_fun, u0, lam=1e-4, beta=1.0):
    """
    Anderson acceleration for fixed point solver adapted from
    http://implicit-layers-tutorial.org/deep_equilibrium_models
    
    Parameters
    ----------
    fp_fun : callable
        function defining fixed point solution, i.e., ``fp_fun(u) = u``
    u0 : array-/Tensor-like
        initial guess for fixed point
    lam : float
        regularization for solving linear system
    beta : float
        mixing parameter
    
    Returns
    -------
    u_conv : array-/Tensor-like
        fixed point solution satisfying ``fp_fun(u_conv) = u_conv``
    
    """
    # handle UHF/RHF
    if u0.dim() == 4:
        nmol, nsp, norb, morb = u0.shape
    else:
        nmol, norb, morb = u0.shape
        nsp = 1
    # init history of solver
    m = SCF_BACKWARD_ANDERSON_HISTSIZE
    X = torch.zeros((nmol, m, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    F = torch.zeros((nmol, m, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    X[:,0], F[:,0] = u0.view(nmol, -1), fp_fun(u0).view(nmol, -1)
    X[:,1], F[:,1] = F[:,0], fp_fun(F[:,0].view_as(u0)).view(nmol, -1)
    # set up linear system
    H = torch.zeros((nmol, m+1, m+1), dtype=u0.dtype, device=u0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros((nmol, m+1, 1), dtype=u0.dtype, device=u0.device)
    y[:,0] = 1
    cond, resid = False, 100.
    # solve iteratively
    for k in range(2, SCF_BACKWARD_ANDERSON_MAXITER):
        n = min(k, m)
        G = F[:,:n] - X[:,:n]
        GTG = torch.bmm(G, G.transpose(1,2))
        E_n = torch.eye(n, dtype=u0.dtype, device=u0.device)
        H[:,1:n+1,1:n+1] = GTG + lam * E_n[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:,1:n+1,0]
        Xold = (1 - beta) * (alpha[:,None] @ X[:,:n])[:,0]
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + Xold
        F[:,k%m] = fp_fun(X[:,k%m].view_as(u0)).view(nmol, -1)
        resid1 = (F[:,k%m] - X[:,k%m]).norm().item()
        resid = resid1 / (1e-5 + F[:,k%m].norm().item())
        cond = resid < SCF_BACKWARD_ANDERSON_TOLERANCE
        if cond: break   # solver converged
    if not cond:
        msg = "Anderson solver in SCF backward did not converge for some molecule(s)"
        if RAISE_ERROR_IF_SCF_BACKWARD_FAILS: raise ValueError(msg)
        warnings.warn(msg)
    u_conv = X[:,k%m].view_as(u0)
    return u_conv
    

# TODO: simplify, clean
class SCF(torch.autograd.Function):
    """
    scf loop
    forward and backward
    check function scf_loop for details
    """
    
    @staticmethod
    def forward(ctx, M, w, gss, gpp, gsp, gp2, hsp,
                nHydro, nHeavy, nOccMO, nmol, molsize,
                maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps_E,
                scf_converger, use_sp2, scf_backward_eps, scf_maxiter,
                occ_mode=0, occ_kT=torch.tensor(0.05), smearing="fermi",
                eps_P=1e-5):
        if scf_converger[0] == 4:
            diis_start = scf_converger[1].get('diis_start', 2)
            diis_max = scf_converger[1].get('diis_max', 8)
            detach_diis = scf_converger[1].get('detach_diis', True)
            compress_rank = scf_converger[1].get('compress_rank', None)
            P, notconverged = scf_diis(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy,
                                nOccMO, nmol, molsize, maskd, mask, idxi, idxj, P, eps_E,
                                scf_maxiter=scf_maxiter, occ_mode=occ_mode, occ_kT=occ_kT,
                                smearing=smearing, eps_P=eps_P, diis_start=diis_start, 
                                diis_max=diis_max, detach_diis=detach_diis,
                                compress_rank=compress_rank)

        elif scf_converger[0] == 0:
            P, notconverged = scf_constmix(M, w, gss, gpp, gsp, gp2, hsp,
                                   nHydro, nHeavy, nOccMO, nmol, molsize,
                                   maskd, mask, idxi, idxj, P, eps_E, sp2=use_sp2,
                                   alpha=scf_converger[1], scf_maxiter=scf_maxiter,
                                   occ_mode=occ_mode, occ_kT=occ_kT, smearing=smearing,
                                   eps_P=eps_P)
        elif scf_converger[0] == 3: # KSA
            P, notconverged = scf_forward3(M, w, gss, gpp, gsp, gp2, hsp,
                                   nHydro, nHeavy, nOccMO, nmol, molsize,
                                   maskd, mask, idxi, idxj, P, eps_E, scf_converger[1], scf_maxiter=scf_maxiter)
        else:
            if scf_converger[0] == 1: # adaptive mixing
                scf_forward = scf_forward1
            elif scf_converger[0] == 2: # adaptive mixing, then pulay
                scf_forward = scf_forward2
            P, notconverged = scf_forward(M, w, gss, gpp, gsp, gp2, hsp, \
                               nHydro, nHeavy, nOccMO, nmol, molsize, \
                               maskd, mask, idxi, idxj, P, eps_E, sp2=use_sp2, scf_maxiter=scf_maxiter)
        
        occ_mode = torch.as_tensor(occ_mode, dtype=int, device=M.device)
        occ_kT = torch.as_tensor(occ_kT, dtype=M.dtype, device=M.device)
        ctx.smearing = smearing
        ctx.save_for_backward(P, M, w, gss, gpp, gsp, gp2, hsp, \
                              nHydro, nHeavy, nOccMO, \
                              maskd, mask, idxi, idxj, notconverged, \
                              atom_molid, pair_molid, occ_mode, occ_kT)
        return P, notconverged
    
    
    @staticmethod
    def backward(ctx, grad_P, grad1):
        """
        custom backward of SCF loop
        
        CURRENTLY DOES NOT SUPPORT DOUBLE-BACKWARD!
        FOR CORRECT SECOND DERIVATIVES OF DENSITY MATRIX, USE DIRECT BACKPROP.
        """
        Pin, M, w, gss, gpp, gsp, gp2, hsp, \
            nHydro, nHeavy, nOccMO, \
            maskd, mask, idxi, idxj, notconverged, \
            atom_molid, pair_molid, \
            occ_mode, occ_kT = ctx.saved_tensors
        nmol = Pin.shape[0]
        molsize = Pin.shape[-1]//4
        grads, gvind, gv = {}, [], []
        for i, st in enumerate([M, w, gss, gpp, gsp, gp2, hsp]):
            if st.requires_grad:
                gv.append(st)
                gvind.append(i+1)
            else:
                grads[i+1] = None
        
        with torch.enable_grad():
            Pin.requires_grad_(True)
            if Pin.dim() == 4:
                get_fock_mat, n_spin = fock_u_batch, 2
            else:
                get_fock_mat, n_spin = fock, 1
            F = get_fock_mat(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            e, v = sym_eig_trunc(F, nHeavy, nHydro, nOccMO)
            D = build_dm(e, v, nOccMO, occ_mode=occ_mode, occ_kT=occ_kT, smearing=ctx.smearing) / n_spin
            Pout = unpack(D, nHeavy, nHydro, F.shape[-1])
        
        ## THIS DOES NOT SUPPORT DOUBLE BACKWARD AT THE MOMENT. MAY AS WELL STOP AUTOGRAD TAPE
        with torch.no_grad():
            def affine_eq(u): return grad_P + agrad(Pout, Pin, grad_outputs=u, retain_graph=True)[0]
            u_init = torch.zeros_like(Pin)  #TODO: better initial guess?
            u = fixed_point_anderson(affine_eq, u_init)
            gradients = agrad(Pout, gv, grad_outputs=u, retain_graph=True)
            for t, i in enumerate(gvind): grads[i] = gradients[t]
            
            if notconverged.any():
                warnings.warn("SCF forward : %d/%d not converged" % (notconverged.sum().item(),nmol))
                cond = notconverged.detach()
                #M, w, gss, gpp, gsp, gp2, hsp
                #M shape(nmol*molsizes*molsize, 4, 4)
                if torch.is_tensor(grads[1]):
                    grads[1] = grads[1].reshape(nmol, molsize*molsize, 4, 4)
                    grads[1][cond] = 0.0
                    grads[1] = grads[1].reshape(nmol*molsize*molsize, 4, 4)
                #w shape (npairs, 10, 10)
                if torch.is_tensor(grads[2]): grads[2][cond[pair_molid]] = 0.0
                #gss, gpp, gsp, gp2, hsp shape (natoms,)
                for i in range(3,8):
                    if torch.is_tensor(grads[i]): grads[i][cond[atom_molid]] = 0.0
        
        return grads[1], grads[2], grads[3], grads[4], grads[5], grads[6], grads[7], \
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None
        
    

class SCF0(SCF):
    @staticmethod
    def backward(ctx, grad_P, grad1):
        # ignore gradient on density matrix and eigenvectors/-values
        return None, None, None, None, None, None, None, \
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None, \
               None, None, None, None, None, None, None, None


# TODO: simplify, clean
def scf_loop(const, molsize, nHeavy, nHydro, nOccMO,
             maskd, mask, atom_molid, pair_molid, idxi, idxj, ni, nj, xij, rij, Z,
             zetas, zetap, uss, upp , gss, gsp, gpp, gp2, hsp, beta, Kbeta=None,
             eps_E=1e-5, P=None, sp2=[False], scf_converger=[0,0.15], eig=False, scf_backward=0,
             scf_backward_eps=1e-2, ivans_beta=False, scf_maxiter=200, 
             occ_mode=0, occ_kT=torch.tensor(0.05), smearing="fermi",
             eps_P=1e-5):
    """
    SCF loop
    # check hcore.py for the details of arguments
    eps : convergence criteria for energy and density matrix
    P : if provided, will be used as initial density matrix in scf loop
    return : F, e, v, P, Hcore, w
    """
    device = xij.device
    nmol = nHeavy.shape[0]
    tore = const.tore
    if const.do_timing: t0 = time.time()
    M, w = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni, nj, xij, 
                 rij, Z, zetas, zetap, uss, upp , gss, gpp, gp2, hsp, beta, 
                 nHeavy, nHydro, nOccMO,
                 Kbeta=Kbeta,
                 ivans_beta=ivans_beta,
                )
    
    if const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        const.timing["Hcore + STO Integrals"].append(t1 - t0)
        t0 = time.time()
    if scf_backward == 2 or (not torch.is_tensor(P)):
        P0 = torch.zeros_like(M)  # density matrix
        P0[maskd[Z>1],0,0] = tore[Z[Z>1]]/4.0
        P0[maskd,1,1] = P0[maskd,0,0]
        P0[maskd,2,2] = P0[maskd,0,0]
        P0[maskd,3,3] = P0[maskd,0,0]
        P0[maskd[Z==1],0,0] = 1.0
        P = P0.reshape(nmol,molsize,molsize,4,4) \
            .transpose(2,3) \
            .reshape(nmol, 4*molsize, 4*molsize)
        if nOccMO.dim() == 2:
            P = torch.stack((0.5*P, 0.5*P), dim=1)
    
    #"""
    #scf_backward == 2, directly backward through scf loop
    #             can't reuse P, so put P=None and initial P above
    #"""
    if scf_backward == 2:
        if sp2[0]:
            warnings.warn('SP2 is not used for direct backpropagation through scf loop')
            sp2[0] = False
        if scf_converger[0] == 0:
            Pconv, notconverged = scf_constmix(M, w, gss, gpp, gsp, gp2, hsp,
                                nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask, 
                                idxi, idxj, P, eps_E, sp2=sp2, alpha=scf_converger[1],
                                backward=True, scf_maxiter=scf_maxiter,
                                occ_mode=occ_mode, occ_kT=occ_kT, smearing=smearing, eps_P=eps_P)
        elif scf_converger[0] == 1:
            Pconv, notconverged = scf_forward1(M, w, gss, gpp, gsp, gp2, hsp,
                                nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask, 
                                idxi, idxj, P, eps_E, sp2=sp2, backward=True, scf_maxiter=scf_maxiter)
        else:
            raise ValueError("""For direct backpropagation through scf,
                                must use constant mixing at this moment\n
                                set scf_converger=[0, alpha] or [1]\n""")
    #scf_backward 1, use recursive formula/implicit autodiff
    if scf_backward == 1:
        scfapply = SCF().apply
    #scf_backward 0: ignore the gradient on density matrix
    elif scf_backward == 0:
        scfapply = SCF0().apply

    # apply_params = {k:pp[k] for k in apply_param_map}
    if scf_backward == 0 or scf_backward == 1:
        Pconv, notconverged = scfapply(M, w, gss, gpp, gsp, gp2, hsp,
                        nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask,
                        atom_molid, pair_molid, idxi, idxj, P, eps_E,
                        scf_converger, sp2, scf_backward_eps, scf_maxiter,
                        occ_mode, occ_kT, smearing, eps_P)
    if notconverged.any():
        nnot = notconverged.type(torch.int).sum().data.item()
        warnings.warn("SCF for %d/%d molecules doesn't converge after %d iterations" % (nnot, nmol, scf_maxiter))
        if RAISE_ERROR_IF_SCF_FORWARD_FAILS:
            raise ValueError("SCF for some the molecules in the batch doesn't converge")

    if const.do_timing:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        const.timing["SCF"].append(t1-t0)
    
    if Pconv.dim() == 4:
        F = fock_u_batch(nmol, molsize, Pconv, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    else:
        F = fock(nmol, molsize, Pconv, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    #
    #return Fock matrix, eigenvalues, density matrix, Hcore,  2 electron 2 center integrals, eigenvectors
    if eig:
        if scf_backward >= 1:
            e, v = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO, eig_only=True)
        else:
            e, v = sym_eig_trunc(F, nHeavy, nHydro, nOccMO, eig_only=True)

#        #get charge of each orbital on each atom -- WHY???
#        charge = torch.zeros(nmol, molsize*4, molsize, device=e.device, dtype=e.dtype)
#        v2 = [x**2 for x in v]
#        norb = 4 * nHeavy + nHydro
#                
#        if F.dim() == 4: # open shell
#            # $$$
#            for i in range(nmol):
#                q1 = v2[i][0,:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
#                q1 = q1 + v2[i][1,:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
#                charge[i,:norb[i],:nHeavy[i]] = q1
#                q2 = v2[i][0,:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
#                q2 = q2 + v2[i][1,:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
#                charge[i,:norb[i],nHeavy[i]:(nHeavy[i]+nHydro[i])] = q2
#            charge = charge / 2
#        else: # closed shell
#            for i in range(nmol):
#                charge[i,:norb[i],:nHeavy[i]] = v2[i][:norb[i],:(4*nHeavy[i])].reshape(norb[i],4,nHeavy[i]).sum(dim=1)
#                charge[i,:norb[i],nHeavy[i]:(nHeavy[i]+nHydro[i])] = v2[i][:norb[i],(4*nHeavy[i]):(4*nHeavy[i]+nHydro[i])]
        charge = None
        if type(v) is tuple:
            v = tuple(v_i.transpose(-1,-2) for v_i in v)
        elif torch.is_tensor(v):
            v = v.transpose(-1,-2)

        return F, e, v, Pconv, Hcore, w, charge, notconverged
    else:
        return F, None, None, Pconv, Hcore, w, None, notconverged
