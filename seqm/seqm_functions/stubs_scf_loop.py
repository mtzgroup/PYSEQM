import torch
from .fock import fock
from .fock_u_batch import fock_u_batch
from .energy import elec_energy
from .SP2 import SP2
from .fermi_q import Fermi_Q
from .G_XL_LR import G
from seqm.seqm_functions.canon_dm_prt import Canon_DM_PRT
from .pack import *
from .diag import sym_eig_trunc, sym_eig_trunc1
import time


debug = False


# "adaptive mixing"
def scf_forward1(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO, 
                 nmol, molsize, maskd, mask, idxi, idxj, P, eps, 
                 sp2=[False], backward=False, scf_maxiter=200):
    """
    adaptive mixing algorithm, see cnvg.f
    """
    raise NotImplementedError("Adaptive mixing seems to be broken!")
    nDirect1 = 2
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    
    k = 0
    if P.dim() == 4:
        get_fock_mat, uhf, n_spin = fock_u_batch, True, 2
    else:
        get_fock_mat, uhf, n_spin = fock, False, 1
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
            if backward:
                Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                                                    nHeavy[notconverged],
                                                    nHydro[notconverged],
                                                    nOccMO[notconverged])[1] / n_spin
            elif sp2[0]:
                Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1] / n_spin
            if backward:
                Pold = P + 0.0 # ???
                P = Pnew + 0.0 # ???
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = Pnew[notconverged]
            F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    if backward: fac_register = []
    while(1):
        if notconverged.any():
            if backward:
                Pnew[notconverged] = sym_eig_trunc1(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1] / n_spin
            elif sp2[0]:
                Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1] / n_spin
            if backward:
                with torch.no_grad():
                    if uhf:
                        f = torch.zeros((P.shape[0], 2, 1, 1), dtype=P.dtype, device=P.device)
                        f[notconverged,0] = torch.sqrt( torch.sum( (   Pnew[notconverged,0].diagonal(dim1=1,dim2=2)
                                               - P[notconverged,0].diagonal(dim1=1,dim2=2) \
                                             )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged,0].diagonal(dim1=1,dim2=2)
                                              - P[notconverged,0].diagonal(dim1=1,dim2=2)*2.0
                                              + Pold[notconverged,0].diagonal(dim1=1,dim2=2)
                                          )**2, dim=1 ) ).reshape(-1,1,1)
                        f[notconverged,1] = torch.sqrt( torch.sum( (   Pnew[notconverged,1].diagonal(dim1=1,dim2=2)
                                               - P[notconverged,1].diagonal(dim1=1,dim2=2) \
                                             )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged,1].diagonal(dim1=1,dim2=2)
                                              - P[notconverged,1].diagonal(dim1=1,dim2=2)*2.0
                                              + Pold[notconverged,1].diagonal(dim1=1,dim2=2)
                                          )**2, dim=1 ) ).reshape(-1,1,1)
                    else:
                        f = torch.zeros((P.shape[0], 1, 1), dtype=P.dtype, device=P.device)
                        f[notconverged] = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                               - P[notconverged].diagonal(dim1=1,dim2=2) \
                                             )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                              - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                              + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                          )**2, dim=1 ) ).reshape(-1,1,1)
                    fac_register.append(f)
            else:
                if uhf:
                    fac_a = torch.sqrt( torch.sum( (   Pnew[notconverged,0].diagonal(dim1=1,dim2=2)
                                           - P[notconverged,0].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged,0].diagonal(dim1=1,dim2=2)
                                           - P[notconverged,0].diagonal(dim1=1,dim2=2)*2.0
                                           + Pold[notconverged,0].diagonal(dim1=1,dim2=2)
                                         )**2, dim=1 ) ).reshape(-1,1,1)
                    fac_b = torch.sqrt( torch.sum( (   Pnew[notconverged,1].diagonal(dim1=1,dim2=2)
                                           - P[notconverged,1].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged,1].diagonal(dim1=1,dim2=2)
                                           - P[notconverged,1].diagonal(dim1=1,dim2=2)*2.0
                                           + Pold[notconverged,1].diagonal(dim1=1,dim2=2)
                                         )**2, dim=1 ) ).reshape(-1,1,1)
                    fac = torch.stack((fac_a,fac_b), dim=1)
                else:
                    fac = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                                  torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                           + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                         )**2, dim=1 ) ).reshape(-1,1,1)
            #
            if backward:
                Pold = P + 0.0  # ???
                P = (1. + fac_register[-1]) * Pnew - fac_register[-1] * P
            else:
                Pold[notconverged] = P[notconverged]
                P[notconverged] = (1. + fac) * Pnew[notconverged] - fac * P[notconverged]
            
            F = get_fock_mat(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
            k = k + 1
            if k >= scf_maxiter: return P, notconverged
        else:
            return P, notconverged


#adaptive mixing, pulay
def scf_forward2(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                 nmol, molsize, maskd, mask, idxi, idxj, P, eps, sp2=[False], scf_maxiter=200):
    """
    adaptive mixing algorithm, see cnvg.f
    combine with pulay converger
    #check mopac for which P is stored: P constructed from fock subroutine, or P from pulay algorithm
    """
    dtype = M.dtype
    device = M.device
    #procedure
    #nDirect1 steps of directly taking new density
    #nAdapt steps of adaptive mixing (start preparing for pulay)
    #nFock-nAdapt steps of directly taking new density
    #pulay

    nDirect1 = 2
    nAdapt = 1
    # number of maximal fock matrixes used
    nFock = 5

    """
    *      Emat is matrix with form
    *      |<E(1)*E(1)>  <E(1)*E(2)> ...   -1.0|
    *      |<E(2)*E(1)>  <E(2)*E(2)> ...   -1.0|
    *      |<E(3)*E(1)>  <E(3)*E(2)> ...   -1.0|
    *      |<E(4)*E(1)>  <E(4)*E(2)> ...   -1.0|
    *      |     .            .      ...     . |
    *      |   -1.0         -1.0     ...    0. |
    *
    *   WHERE <E(I)*E(J)> IS THE SCALAR PRODUCT OF [F*P] FOR ITERATION I
    *   TIMES [F*P] FOR ITERATION J.
    """
    # F*P - P*F = [F*P]
    FPPF = torch.zeros(nmol, nFock, molsize*4, molsize*4, dtype=dtype,device=device)
    EMAT = (torch.eye(nFock+1, nFock+1, dtype=dtype, device=device) - 1.0).expand(nmol,nFock+1,nFock+1).tril().clone()
    EVEC = torch.zeros_like(EMAT) # EVEC is <E(i)*E(j)> scaled by a constant
    FOCK = torch.zeros_like(FPPF) # store last n=nFock number of Fock matrixes
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    k = 0
    if P.dim() == 4:
        raise RuntimeError("Adaptive plus Pulay mixing doesn't support UHF yet")
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
            if sp2[0]:
                Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                                                   nHeavy[notconverged],
                                                   nHydro[notconverged],
                                                   nOccMO[notconverged])[1]
            Pold[notconverged] = P[notconverged]
            P[notconverged] = Pnew[notconverged]
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    
    
    """
    cFock = cFock + 1 if cFock < nFock else nFock
    #store fock matrix
    FOCK[...,counter,:,:] = F
    FPPF[...,counter,:,:] = (F.matmul(P) - P.matmul(F)).triu()
    # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
    #only compute lower triangle as Emat are symmetric
    EMAT[...,counter,:cFock] = torch.sum(FPPF[..., counter:(counter+1),:,:]*FPPF[...,:cFock,:,:], dim=(2,3))
    """
    for i in range(nAdapt):
        if notconverged.any():
            if sp2[0]:
                Pnew[notconverged] = unpack(
                                            SP2(
                                                pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                                nOccMO[notconverged], sp2[1]
                                                ),
                                            nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                Pnew[notconverged] = sym_eig_trunc(F[notconverged],
                             nHeavy[notconverged], nHydro[notconverged],
                             nOccMO[notconverged])[1]
            fac = torch.sqrt( torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2) \
                                         )**2, dim=1 ) / \
                              torch.sum( (   Pnew[notconverged].diagonal(dim1=1,dim2=2)
                                           - P[notconverged].diagonal(dim1=1,dim2=2)*2.0
                                           + Pold[notconverged].diagonal(dim1=1,dim2=2)
                                       )**2, dim=1 ) ).reshape(-1,1,1)
            Pold[notconverged] = P[notconverged]
            P[notconverged] = (1.0 + fac) * Pnew[notconverged] - fac * P[notconverged]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
            k = k + 1
        else:
            return P, notconverged
    del Pold, Pnew

    #start prepare for pulay algorithm
    counter = -1 # index of stored FPPF for current iteration: 0, 1, ..., cFock-1
    cFock = 0 # in current iteraction, number of fock matrixes stored, cFock <= nFock
    #Pulay algorithm needs at least two previous stored density and Fock matrixes to start
    while (cFock<2):
        if notconverged.any():
            cFock = cFock + 1 if cFock < nFock else nFock
            #store fock matrix
            counter = (counter + 1)%nFock
            FOCK[notconverged, counter, :, :] = F[notconverged]
            FPPF[notconverged, counter, :, :] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            #only compute lower triangle as Emat are symmetric
            EMAT[notconverged, counter, :cFock] = torch.sum(FPPF[notconverged, counter:(counter+1),:,:] * FPPF[notconverged,:cFock,:,:], dim=(2,3))
            if sp2[0]:
                P[notconverged] = unpack(
                                         SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                             ),
                                         nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                P[notconverged] = sym_eig_trunc(F[notconverged],
                                                nHeavy[notconverged],
                                                nHydro[notconverged],
                                                nOccMO[notconverged])[1]
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            if debug:
                print("scf ", k, torch.max(err).item(), torch.sum(notconverged).item())
            k = k + 1
        else:
            return P, notconverged
    
    #start pulay algorithm
    while(1):
        if notconverged.any():
            EVEC[notconverged] = EMAT[notconverged] + EMAT[notconverged].tril(-1).transpose(1,2)
            # work-around for in-place operation (needed? more elegant solution?)
            EVcF = EVEC[notconverged,:cFock,:cFock].clone()
            EVnorm = EVEC[notconverged,counter:(counter+1),counter:(counter+1)].clone()
            EVEC[notconverged,:cFock,:cFock] = EVcF / EVnorm
            coeff = -torch.inverse(EVEC[notconverged,:(cFock+1),:(cFock+1)])[...,:-1,-1]
            F[notconverged] = torch.sum(FOCK[notconverged,:cFock,:,:]*coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
            if sp2[0]:
                P[notconverged] = unpack(
                                         SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                             ),
                                         nHeavy[notconverged], nHydro[notconverged], 4*molsize)
            else:
                P[notconverged] = sym_eig_trunc(F[notconverged],
                                                nHeavy[notconverged],
                                                nHydro[notconverged],
                                                nOccMO[notconverged])[1]
            #
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            
            cFock = cFock + 1 if cFock < nFock else nFock
            counter = (counter + 1)%nFock
            FOCK[notconverged,counter,:,:] = F[notconverged]
            FPPF[notconverged,counter,:,:] = (F[notconverged].matmul(P[notconverged]) - P[notconverged].matmul(F[notconverged])).triu()
            # in mopac7 interp.f pulay subroutine, only lower triangle of FPPF is used to construct EMAT
            #only compute lower triangle as Emat are symmetric
            EMAT[notconverged,counter,:cFock] = torch.sum(FPPF[notconverged, counter:(counter+1),:,:]*FPPF[notconverged,:cFock,:,:], dim=(2,3))
            
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged]-Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            max_err = torch.max(err).item()
            Nnot = torch.sum(notconverged).item()
            if debug: print("scf ", k, max_err, Nnot)
            k = k + 1
            if k >= scf_maxiter: return P, notconverged
        else:
            return P, notconverged

        
def scf_forward3(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO, 
                 nmol, molsize, maskd, mask, idxi, idxj, P, eps, 
                 xl_bomd_params, backward=False):
    """
    DM scf optimization using KSA
    $$$ probably, not properly optimized for batches. 
    backward is for testing purpose, default is False
    if want to test scf backward directly through the loop in this function, turn backward to be True
    """
    err = torch.ones(nmol, dtype=P.dtype, device=P.device)
    notconverged = torch.ones(nmol,dtype=torch.bool, device=M.device)
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
         .transpose(2,3) \
         .reshape(nmol, 4*molsize, 4*molsize)
    
    Temp = xl_bomd_params['T_el']
    kB = 8.61739e-5 # eV/K, kB = 6.33366256e-6 Ry/K, kB = 3.166811429e-6 Ha/K, #kB = 3.166811429e-6 #Ha/K
    SCF_err = torch.tensor([1.0], dtype=P.dtype, device=P.device)
    COUNTER = 0
    
    Rank = xl_bomd_params['max_rank']
    V = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device)
    W = torch.zeros((P.shape[0], P.shape[1], P.shape[2], Rank), dtype=P.dtype, device=P.device)
    
    K0 = 1.0
    if P.dim() == 4:
        raise RuntimeError("Krylov Subspace SCF solver doesn't support UHF yet")
    F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
    D, S_Ent, QQ, e, Fe_occ, mu0, Occ_mask = Fermi_Q(F, Temp, nOccMO, nHeavy, nHydro, kB, scf_backward=0)
    dDS = K0 * (D - P)
    dW = dDS
    
    Eelec = torch.zeros((nmol), dtype=P.dtype, device=P.device)
    Eelec_new = torch.zeros_like(Eelec)
    if debug: print("step, DM rmse, dE, number of not converged")
    
    while (1):
        start_time = time.time()
        if notconverged.any():
            COUNTER = COUNTER + 1
            D[notconverged], S_Ent[notconverged], QQ[notconverged], e[notconverged], \
                Fe_occ[notconverged], mu0[notconverged], Occ_mask[notconverged] = \
                Fermi_Q(F[notconverged], Temp, nOccMO[notconverged], nHeavy[notconverged], 
                        nHydro[notconverged], kB, scf_backward = 0)
            dDS = K0 * (D - P)
            dW = dDS
            k = -1
            Error = torch.tensor([10], dtype=D.dtype, device=D.device)
            while k < Rank-1 and torch.max(Error) > xl_bomd_params['err_threshold']:
                k = k + 1
                V[:,:,:,k] = dW
                for j in range(0,k): #Orthogonalized Krylov vectors (Arnoldi)
                    V[:,:,:,k] = V[:,:,:,k] - torch.sum(V[:,:,:,k].transpose(1,2) * V[:,:,:,j], dim=(1,2)).view(-1, 1, 1) * V[:,:,:,j]
                V[:,:,:,k] = V[:,:,:,k] / torch.sqrt(torch.sum(V[:,:,:,k].transpose(1,2) * V[:,:,:,k], dim=(1,2))).view(-1, 1, 1)
                d_D = V[:,:,:,k]
                FO1 = G(nmol, molsize, d_D, M, maskd, mask, idxi, idxj, w, \
                        gss=gss, gpp=gpp, gsp=gsp, gp2=gsp, hsp=gsp)
                
                PO1 = Canon_DM_PRT(FO1, Temp, nHeavy, nHydro, QQ, e, mu0, 8, kB, Occ_mask)
                W[:,:,:,k] = K0 * (PO1 - V[:,:,:,k])
                dW = W[:,:,:,k]
                Rank_m = k + 1
                O = torch.zeros((D.shape[0], Rank_m, Rank_m), dtype=D.dtype, device=D.device)
                for I in range(0, Rank_m):
                    for J in range(I, Rank_m):
                        O[:,I,J] = torch.sum(W[:,:,:,I].transpose(1,2) * W[:,:,:,J], dim=(1,2))
                        O[:,J,I] = O[:,I,J]
                
                MM = torch.inverse(O)
                IdentRes = torch.zeros(D.shape, dtype=D.dtype, device=D.device)
                for I in range(0,Rank_m):
                    for J in range(0, Rank_m):
                        IdentRes = IdentRes + \
                            MM[:,I,J].view(-1, 1, 1) * torch.sum(W[:,:,:,J].transpose(1,2) * dDS, dim=(1,2)).view(-1, 1, 1) * W[:,:,:,I]
                Error = torch.linalg.norm(IdentRes - dDS, ord='fro', dim=(1,2))/torch.linalg.norm(dDS, ord='fro', dim=(1,2))
            
            for I in range(0, Rank_m):
                for J in range(0, Rank_m):
                    P[notconverged] = P[notconverged] - \
                            MM[notconverged,I,J].view(-1, 1, 1) *torch.sum(W[notconverged,:,:,J].transpose(1,2)*dDS[notconverged], dim=(1,2)).view(-1, 1, 1) * V[notconverged,:,:,I]
            
            F = fock(nmol, molsize, P, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp)
            Eelec_new[notconverged] = elec_energy(P[notconverged], F[notconverged], Hcore[notconverged])
            err[notconverged] = torch.abs(Eelec_new[notconverged] - Eelec[notconverged])
            Eelec[notconverged] = Eelec_new[notconverged]
            notconverged = err > eps
            SCF_err = torch.linalg.norm(dDS[notconverged], ord='fro', dim=(1,2))
            if debug:
                end_time = time.time()
                print(COUNTER, SCF_err.cpu().numpy(), err.cpu().numpy(), torch.sum(notconverged).item(), end_time - start_time)
            if COUNTER >= scf_maxiter: return P, notconverged
        else:
            return P, notconverged
        

