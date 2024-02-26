import torch
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .pack import *
from .diag import sym_eig_trunc, sym_eig_trunc1

debug = False


#@torch.jit.script
def adapt_mix_fac_UHF(P_old, P, P_new):
    f_a = torch.sqrt( torch.sum( (  P_new[:,0].diagonal(dim1=1,dim2=2)
                                  - P[:,0].diagonal(dim1=1,dim2=2) \
                                 )**2, dim=1 ) / \
                      torch.sum( (  P_new[:,0].diagonal(dim1=1,dim2=2)
                                  - P[:,0].diagonal(dim1=1,dim2=2) * 2.0
                                  + P_old[:,0].diagonal(dim1=1,dim2=2)
                                 )**2, dim=1 ) ).reshape(-1,1,1)
    f_b = torch.sqrt( torch.sum( (  P_new[:,1].diagonal(dim1=1,dim2=2)
                                  - P[:,1].diagonal(dim1=1,dim2=2) \
                                 )**2, dim=1 ) / \
                      torch.sum( (  P_new[:,1].diagonal(dim1=1,dim2=2)
                                  - P[:,1].diagonal(dim1=1,dim2=2) * 2.0
                                  + P_old[:,1].diagonal(dim1=1,dim2=2)
                                 )**2, dim=1 ) ).reshape(-1,1,1)
    f = torch.stack((f_a, f_b), dim=1)
    return f

#@torch.jit.script
def adapt_mix_fac_RHF(P_old, P, P_new):
    f = torch.sqrt( torch.sum( (  P_new.diagonal(dim1=1,dim2=2)
                                - P.diagonal(dim1=1,dim2=2) \
                               )**2, dim=1 ) / \
                    torch.sum( (  P_new.diagonal(dim1=1,dim2=2)
                                - P.diagonal(dim1=1,dim2=2) * 2.0
                                + P_old.diagonal(dim1=1,dim2=2)
                               )**2, dim=1 ) ).reshape(-1,1,1)
    return f



#RHF: adaptive mixing
#@torch.jit.script
def scf_forward1(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                 nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                 dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for i in range(nDirect1):
        if notconverged.any():
            Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                     nHydro[notconverged], nOccMO[notconverged])[1]
            Pold[notconverged] = P[notconverged] + 0.
            P[notconverged] = Pnew[notconverged] + 0.
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
        else:
            return P, notconverged
    for i in range(nDirect1, max_scf_iter+1):
        Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1]
        f = adapt_mix_fac_RHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
        Pold[notconverged] = P[notconverged] + 0.
        P[notconverged] = (1. + f) * Pnew[notconverged] - f * P[notconverged]
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        max_err = torch.max(err).item()
        Nnot = torch.sum(notconverged).item()
        if debug: print("scf ", k, max_err, Nnot)
        if not notconverged.any(): break
    return P, notconverged


# RHF: adaptive mixing + SP2
#@torch.jit.script
def scf_forward1_sp2(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                     nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                     dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for i in range(nDirect1):
        if notconverged.any():
            Pnew[notconverged] = unpack( SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            Pold[notconverged] = P[notconverged] + 0.
            P[notconverged] = Pnew[notconverged] + 0.
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
        else:
            return P, notconverged
    for i in range(nDirect1, max_scf_iter+1):
        Pnew[notconverged] = unpack( SP2(
                                         pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                         nOccMO[notconverged], sp2[1]
                                        ),
                                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
        f = adapt_mix_fac_RHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
        Pold[notconverged] = P[notconverged] + 0.
        P[notconverged] = (1. + f) * Pnew[notconverged] - f * P[notconverged]
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        max_err = torch.max(err).item()
        Nnot = torch.sum(notconverged).item()
        if debug: print("scf ", k, max_err, Nnot)
        if not notconverged.any(): break
    return P, notconverged

# UHF: adaptive mixing
#@torch.jit.script
def scf_forward1_u(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                   nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                   dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for i in range(nDirect1):
        if notconverged.any():
            Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                     nHydro[notconverged], nOccMO[notconverged])[1] / 2
            Pold[notconverged] = P[notconverged] + 0.
            P[notconverged] = Pnew[notconverged] + 0.
            F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
        else:
            return P, notconverged
    for i in range(nDirect1, max_scf_iter+1):
        Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1] / 2
        f = adapt_mix_fac_UHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
        Pold[notconverged] = P[notconverged] + 0.
        P[notconverged] = (1. + f) * Pnew[notconverged] - f * P[notconverged]
        F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        max_err = torch.max(err).item()
        Nnot = torch.sum(notconverged).item()
        if debug: print("scf ", k, max_err, Nnot)
        if not notconverged.any(): break
    return P, notconverged


# UHF: adaptive mixing + SP2
#@torch.jit.script
def scf_forward1_sp2_u(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                       nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                       dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for i in range(nDirect1):
        if notconverged.any():
            Pnew[notconverged] = unpack( SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            Pold[notconverged] = P[notconverged] + 0.
            P[notconverged] = Pnew[notconverged] + 0.
            F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
        else:
            return P, notconverged
    for i in range(nDirect1, max_scf_iter+1):
        Pnew[notconverged] = unpack( SP2(
                                         pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                         nOccMO[notconverged], sp2[1]
                                        ),
                                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
        f = adapt_mix_fac_UHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
        Pold[notconverged] = P[notconverged] + 0.
        P[notconverged] = (1. + f) * Pnew[notconverged] - f * P[notconverged]
        F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        max_err = torch.max(err).item()
        Nnot = torch.sum(notconverged).item()
        if debug: print("scf ", k, max_err, Nnot)
        if not notconverged.any(): break
    return P, notconverged


# adaptive mixing: direct backprop
def scf_forward1_bwd(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                     nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                     dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)

    if P.dim() == 4:
        get_fock_mat, uhf, div = fock_u_batch, True, 2
        get_adapt_mixin, f_shape = adapt_mix_fac_UHF, (2,1,1)
    else:
        get_fock_mat, uhf, div = fock, False, 1
        get_adapt_mixin, f_shape = adapt_mix_fac_RHF, (1,1)
    F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    Pnew = torch.zeros_like(P)
    Pold = torch.zeros_like(P)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for i in range(nDirect1):
        if notconverged.any():
            Pnew[notconverged] = sym_eig_trunc1(F[notconverged], nHeavy[notconverged],
                                     nHydro[notconverged], nOccMO[notconverged])[1] / div
            Pold = P + 0.
            P = Pnew + 0.
            F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
        else:
            return P, notconverged

    fac_register = []
    for i in range(nDirect1, max_scf_iter+1):
        Pnew[notconverged] = sym_eig_trunc1(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1] / div
        with torch.no_grad():
            f = torch.zeros((P.shape[0], *f_shape), dtype=P.dtype, device=P.device)
            f[notconverged] = get_adapt_mixin(Pold[notconverged], P[notconverged],
                                              Pnew[notconverged])
            fac_register.append(f)
        Pold = P + 0.0  # ???
        P = (1. + fac_register[-1]) * Pnew - fac_register[-1] * P
        F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        max_err = torch.max(err).item()
        Nnot = torch.sum(notconverged).item()
        if debug: print("scf ", k, max_err, Nnot)
        if not notconverged.any(): break
    return P, notconverged
    

