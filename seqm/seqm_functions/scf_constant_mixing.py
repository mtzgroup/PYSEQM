import torch
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .pack import *
from .diag import sym_eig_trunc, sym_eig_trunc1


debug = False



# RHF: constant mixing
#@torch.jit.script
def scf_forward0(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                 nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                 alpha: float, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(max_scf_iter):
#        start_time = time.time()
        Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1]
        P[notconverged] = alpha * P[notconverged] + (1. - alpha) * Pnew[notconverged]
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
#            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())#,  end_time-start_time )
        if not notconverged.any(): break
    return P, notconverged

#tr_scf_forward0 = torch.jit.trace()


# RHF: constant mixing + SP2
#@torch.jit.script
def scf_forward0_sp2(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                     nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                     alpha: float, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(max_scf_iter):
#        start_time = time.time()
        Pnew[notconverged] = unpack( SP2(
                                         pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                         nOccMO[notconverged], sp2[1]
                                        ),
                                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
        P[notconverged] = alpha * P[notconverged] + (1. - alpha) * Pnew[notconverged]
        F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
#            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())#,  end_time-start_time )
        if not notconverged.any(): break
    return P, notconverged


# UHF: constant mixing
#@torch.jit.script
def scf_forward0_u(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                   nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                   alpha: float, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(max_scf_iter):
#        start_time = time.time()
        Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                           nHydro[notconverged], nOccMO[notconverged])[1] / 2
        P[notconverged] = alpha * P[notconverged] + (1. - alpha) * Pnew[notconverged]
        F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
#            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())#,  end_time-start_time )
        if not notconverged.any(): break
    return P, notconverged


# UHF: constant mixing + SP2
#@torch.jit.script
def scf_forward0_sp2_u(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                       nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                       alpha: float, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(max_scf_iter):
#        start_time = time.time()
        Pnew[notconverged] = unpack( SP2(
                                         pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                         nOccMO[notconverged], sp2[1]
                                        ),
                                    nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
        P[notconverged] = alpha * P[notconverged] + (1. - alpha) * Pnew[notconverged]
        F = fock_u_batch(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
#            end_time = time.time()
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())#,  end_time-start_time )
        if not notconverged.any(): break
    return P, notconverged



# constant mixing: direct backprop
#@torch.jit.script
def scf_forward0_bwd(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                     nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                     alpha: float, sp2=[False, 1e-4], max_scf_iter: int=200):
    """
    alpha : mixing parameters, alpha=0.0, directly take the new density matrix
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    if P.dim() == 4:
        get_fock_mat, div = fock_u_batch, 2
    else:
        get_fock_mat, div = fock, 1
    Pnew = torch.zeros_like(P)
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    Hcore = M.reshape(nmol, molsize, molsize, 4, 4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)
    Eelec = elec_energy(P, F, Hcore)
    Eelec_new = torch.zeros_like(Eelec)
    for k in range(max_scf_iter):
        Pnew[notconverged] = sym_eig_trunc1(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1] / 2
        Pold = P + 0.
        P = alpha * P + (1. - alpha) * Pnew
        F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
        Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
        err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
        Eelec[notconverged] = Eelec_new[notconverged]
        notconverged = err > eps
        if debug:
            print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
        if not notconverged.any(): break
    return P, notconverged

