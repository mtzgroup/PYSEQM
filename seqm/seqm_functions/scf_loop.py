import torch
import warnings
#import time
from typing import Callable
from torch.autograd import grad as agrad
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .diag import sym_eig_trunc, sym_eig_trunc1
from .scf_constant_mixing import *
from .scf_adaptive_mixing import (scf_forward1, scf_forward1_sp2,
                                  scf_forward1_u, scf_forward1_sp2_u,
                                  scf_forward1_bwd)
from .scf_adaptive_plus_pulay import *
from .scf_KSA import *

#from .check import check
#scf_backward==0: ignore the gradient on density matrix
#scf_backward==1: use recursive formula/implicit autodiff
#scf_backward==2: go backward scf loop directly

MAX_ITER = 2000
RAISE_ERROR_IF_SCF_FORWARD_FAILS = False
#if true, raise error rather than ignore those non-convered molecules

# tolerance, max no. iteration, and history size for Anderson acceleration
# in solving fixed point problem in implicit autodiff
SCF_BACKWARD_ANDERSON_TOLERANCE = 1e-4  # this seems stable enough, but TODO!
SCF_BACKWARD_ANDERSON_MAXITER = 50      # sufficient for all test cases
SCF_BACKWARD_ANDERSON_HISTSIZE = 5      # seems reasonable, but TODO!




        
#@torch.jit.script #(~> callable argument or function definition not supported)
def fixed_point_anderson(fp_fun: Callable, u0: torch.Tensor, lam: float=1e-4, beta: float=1.0,
                         thresh: float=1e-4, hist_size : int=5, max_iter: int=50):
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
    thresh : float
        convergence threshold
    hist_size : int
        history size for Anderson acceleration
    max_iter : int
        maximum number of iterations for solving LSE
    
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
    X = torch.zeros((nmol, hist_size, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    F = torch.zeros((nmol, hist_size, nsp*norb*morb), dtype=u0.dtype, device=u0.device)
    X[:,0], F[:,0] = u0.view(nmol, -1), fp_fun(u0).view(nmol, -1)
    X[:,1], F[:,1] = F[:,0], fp_fun(F[:,0].view_as(u0)).view(nmol, -1)
    # set up linear system
    H = torch.zeros((nmol, hist_size+1, hist_size+1), dtype=u0.dtype, device=u0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros((nmol, hist_size+1, 1), dtype=u0.dtype, device=u0.device)
    y[:,0] = 1
    cond, resid = False, 100.
    # solve iteratively
    for k in range(2, max_iter+2):
        n = min(k, hist_size)
        G = F[:,:n] - X[:,:n]
        GTG = torch.bmm(G, G.transpose(1,2))
        E_n = torch.eye(n, dtype=u0.dtype, device=u0.device)
        H[:,1:n+1,1:n+1] = GTG + lam * E_n[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:,1:n+1,0]
        Xold = (1 - beta) * (alpha[:,None] @ X[:,:n])[:,0]
        X[:,k%hist_size] = beta * (alpha[:,None] @ F[:,:n])[:,0] + Xold
        F[:,k%hist_size] = fp_fun(X[:,k%hist_size].view_as(u0)).view(nmol, -1)
        resid1 = (F[:,k%hist_size] - X[:,k%hist_size]).norm()
        resid = resid1 / (1e-5 + F[:,k%hist_size].norm())
        cond = resid < thresh
        if cond: break   # solver converged
    if not cond:
        msg = "Anderson solver in SCF backward did not converge for some molecule(s)"
        warnings.warn(msg)
    u_conv = X[:,k%hist_size].view_as(u0)
    return u_conv
    

class SCF(torch.autograd.Function):
    """
    scf loop
    forward and backward
    check function scf_loop for details
    """
    def __init__(self, scf_converger=[2], use_sp2=[False, 1e-4],
                 scf_backward_eps: float=1.0e-2, is_uhf: bool=False):
        SCF.sp2 = use_sp2
        SCF.converger = scf_converger
        SCF.scf_backward_eps = scf_backward_eps
        scf_solver_str = "scf_forward"+str(scf_converger[0])
        if use_sp2[0]:
            scf_solver_str += "_sp2"
        else:
            SCF.sp2 = [False, 1e-4]
        if is_uhf: scf_solver_str += "_u"
        SCF.scf_routine = eval(scf_solver_str)
        SCF.scf_opt = None if scf_converger[0]==2 else scf_converger[1]
    
    
    @staticmethod
    def forward(ctx, M, w, gss, gpp, gsp, gp2, hsp,
                nHydro, nHeavy, nOccMO, nmol: int, molsize: int,
                maskd, mask, atom_molid, pair_molid, idxi, idxj, P, eps: float):
        P, notconverged = SCF.scf_routine(M, w, gss, gpp, gsp, gp2, hsp,
                                  nHydro, nHeavy, nOccMO, nmol, molsize,
                                  maskd, mask, idxi, idxj, P, eps, SCF.scf_opt,
                                  sp2=SCF.sp2, max_scf_iter=MAX_ITER)
        eps = torch.as_tensor(eps, dtype=M.dtype, device=M.device)
        ctx.save_for_backward(P, M, w, gss, gpp, gsp, gp2, hsp, \
                              nHydro, nHeavy, nOccMO, \
                              maskd, mask, idxi, idxj, eps, notconverged, \
                              atom_molid, pair_molid)
        
        return P, notconverged
    
    
    @staticmethod
    def backward(ctx, grad_P, grad1):
        """
        custom backward of SCF loop
        
        CURRENTLY DOES NOT SUPPORT DOUBLE-BACKWARD!
        (irrelevant for dE^2/dx^2 or dy/dparam, if y is no derivative itself like forces)
        FOR CORRECT SECOND DERIVATIVES OF DENSITY MATRIX, USE DIRECT BACKPROP.
        """
        #TODO: clean up when fully switching to implicit autodiff
        Pin, M, w, gss, gpp, gsp, gp2, hsp, \
        nHydro, nHeavy, nOccMO, \
        maskd, mask, idxi, idxj, eps, notconverged, \
        atom_molid, pair_molid = ctx.saved_tensors
        nmol = Pin.shape[0]
        molsize = Pin.shape[-1]//4
        grads, gvind = {}, []
        gv = []
        for i, st in enumerate([M, w, gss, gpp, gsp, gp2, hsp]):
            if st.requires_grad:
                gv.append(st)
                gvind.append(i+1)
            else:
                grads[i+1] = None
        with torch.enable_grad():
            Pin.requires_grad_(True)
            if Pin.dim() == 4:
                F = fock_u_batch(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
                Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1] / 2
            else:
                F = fock(nmol, molsize, Pin, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
                Pout = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO)[1]
        
        ## THIS DOES NOT SUPPORT DOUBLE BACKWARD. MAY AS WELL STOP AUTOGRAD TAPE
        with torch.no_grad():
            def affine_eq(u): return grad_P + agrad(Pout, Pin, grad_outputs=u, retain_graph=True)[0]
            u_init = torch.zeros_like(Pin)  #TODO: better initial guess?
            u = fixed_point_anderson(affine_eq, u_init, hist_size=SCF_BACKWARD_ANDERSON_HISTSIZE,
                                     max_iter=SCF_BACKWARD_ANDERSON_MAXITER,
                                     thresh=SCF_BACKWARD_ANDERSON_TOLERANCE)
            gradients = agrad(Pout, gv, grad_outputs=u, retain_graph=True)
            for t, i in enumerate(gvind): grads[i] = gradients[t]
            
#        with torch.no_grad():
            if notconverged.any():
                warnings.warn("SCF for/back-ward : %d/%d not converged" % (notconverged.sum().item(),nmol))
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
               None, None, None, None, None, None, None, None
        
    

class SCF0(SCF):
    @staticmethod
    def backward(ctx, grad_P, grad1):
        # ignore gradient on density matrix and eigenvectors/-values
        return None, None, None, None, None, None, None, \
               None, None, None, \
               None, None, \
               None, None, None, None, None, None, None, None


def scf_loop(const, molsize: int, nHeavy, nHydro, nOccMO, \
             maskd, mask, atom_molid, pair_molid, idxi, idxj, ni, nj, xij, rij, Z, \
             zetas, zetap, uss, upp , gss, gsp, gpp, gp2, hsp, beta, Kbeta=None, \
             eps: float=1e-4, P=None, sp2=[False, 1e-4], scf_converger=[1],
             eig: bool=False, scf_backward: int=0, scf_backward_eps: float=1e-2):
    """
    SCF loop
    # check hcore.py for the details of arguments
    eps : convergence criteria for density matrix on density matrix
    P : if provided, will be used as initial density matrix in scf loop
    return : F, e, P, Hcore, w, v
    """
    device = xij.device
    nmol = nHeavy.shape[0]
    tore = const.tore
#    if const.do_timing: t0 = time.time()
    M, w = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni, nj, xij, 
                 rij, Z, zetas, zetap, uss, upp , gss, gpp, gp2, hsp, beta, 
                 Kbeta=Kbeta)
    
#    if const.do_timing:
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        t1 = time.time()
#        const.timing["Hcore + STO Integrals"].append(t1 - t0)
#        t0 = time.time()
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
    
    uhf = P.dim() == 4
    #scf_backward == 2, directly backward through scf loop
    #             can't reuse P, so put P=None and initial P above
    if scf_backward == 2:
        eig_decomp = sym_eig_trunc1
        if sp2[0]:
            warnings.warn('SP2 is not used for direct backpropagation through scf loop')
            sp2[0] = False
        if scf_converger[0] == 0:
            Pconv, notconverged = scf_forward0_bwd(M, w, gss, gpp, gsp, gp2, hsp,
                                nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask, 
                                idxi, idxj, P, eps, scf_converger[1], sp2=sp2,
                                max_scf_iter=MAX_ITER)
        elif scf_converger[0] == 1:
            Pconv, notconverged = scf_forward1_bwd(M, w, gss, gpp, gsp, gp2, hsp,
                                nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask, 
                                idxi, idxj, P, eps, 0., sp2=sp2, max_scf_iter=MAX_ITER)
        else:
            raise ValueError("""For direct backpropagation through scf,
                                must use constant mixing at this moment\n
                                set scf_converger=[0, alpha] or [1]\n""")
    #scf_backward 1, use implicit autodiff
    elif scf_backward == 1:
        eig_decomp = sym_eig_trunc1
        scfapply = SCF(use_sp2=sp2, scf_converger=scf_converger,
                       scf_backward_eps=scf_backward_eps, is_uhf=uhf).apply
        Pconv, notconverged = scfapply(M, w, gss, gpp, gsp, gp2, hsp,
                        nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask,
                        atom_molid, pair_molid, idxi, idxj, P, eps)
    #scf_backward 0: ignore the gradient on density matrix
    elif scf_backward == 0:
        eig_decomp = sym_eig_trunc
        scfapply = SCF0(use_sp2=sp2, scf_converger=scf_converger, is_uhf=uhf).apply
        Pconv, notconverged = scfapply(M, w, gss, gpp, gsp, gp2, hsp,
                        nHydro, nHeavy, nOccMO, nmol, molsize, maskd, mask,
                        atom_molid, pair_molid, idxi, idxj, P, eps)
    
    if notconverged.any():
        nnot = notconverged.type(torch.int).sum().data.item()
        warnings.warn("SCF for %d/%d molecules doesn't converge after %d iterations" % (nnot, nmol, MAX_ITER))
        if RAISE_ERROR_IF_SCF_FORWARD_FAILS:
            raise ValueError("SCF for some the molecules in the batch doesn't converge")

#    if const.do_timing:
#        if torch.cuda.is_available(): torch.cuda.synchronize()
#        t1 = time.time()
#        const.timing["SCF"].append(t1-t0)
    
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
        e, v = eig_decomp(F, nHeavy, nHydro, nOccMO, eig_only=True)
#        if scf_backward >= 1:
#            e, v = sym_eig_trunc1(F, nHeavy, nHydro, nOccMO, eig_only=True)
#        else:
#            e, v = sym_eig_trunc(F, nHeavy, nHydro, nOccMO, eig_only=True)

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
        return F, e, Pconv, Hcore, w, charge, notconverged
    else:
        return F, None, Pconv, Hcore, w, None, notconverged
