import torch
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .pack import *
from .diag import sym_eig_trunc, sym_eig_trunc1
from .scf_adaptive_mixing import adapt_mix_fac_RHF, adapt_mix_fac_UHF
import warnings
#import time


debug = False


#@torch.jit.script
def scf_forward2(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                 nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                 dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
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
            Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                     nHydro[notconverged], nOccMO[notconverged])[1]
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
            Pnew[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                     nHydro[notconverged], nOccMO[notconverged])[1]
            fac = adapt_mix_fac_RHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
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
            P[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1]
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
    while (1):
        if notconverged.any():
            EVEC[notconverged] = EMAT[notconverged] + EMAT[notconverged].tril(-1).transpose(1,2)
            # work-around for in-place operation (needed? more elegant solution?)
            EVcF = EVEC[notconverged,:cFock,:cFock].clone()
            EVnorm = EVEC[notconverged,counter:(counter+1),counter:(counter+1)].clone()
            EVEC[notconverged,:cFock,:cFock] = EVcF / EVnorm
            coeff = -torch.inverse(EVEC[notconverged,:(cFock+1),:(cFock+1)])[...,:-1,-1]
            F[notconverged] = torch.sum(FOCK[notconverged,:cFock,:,:]*coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
            P[notconverged] = sym_eig_trunc(F[notconverged], nHeavy[notconverged],
                                 nHydro[notconverged], nOccMO[notconverged])[1]
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
            if k >= max_scf_iter: return P, notconverged
        else:
            return P, notconverged
    

#@torch.jit.script
def scf_forward2_sp2(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO,
                     nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float,
                     dummy, sp2=[False, 1e-4], max_scf_iter: int=200):
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
            Pnew[notconverged] = unpack( SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
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
            Pnew[notconverged] = unpack( SP2(
                                             pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                             nOccMO[notconverged], sp2[1]
                                            ),
                                        nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
            fac = adapt_mix_fac_RHF(Pold[notconverged], P[notconverged], Pnew[notconverged])
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
            P[notconverged] = unpack( SP2(
                                          pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                          nOccMO[notconverged], sp2[1]
                                         ),
                                     nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
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
    while (1):
        if notconverged.any():
            EVEC[notconverged] = EMAT[notconverged] + EMAT[notconverged].tril(-1).transpose(1,2)
            # work-around for in-place operation (needed? more elegant solution?)
            EVcF = EVEC[notconverged,:cFock,:cFock].clone()
            EVnorm = EVEC[notconverged,counter:(counter+1),counter:(counter+1)].clone()
            EVEC[notconverged,:cFock,:cFock] = EVcF / EVnorm
            coeff = -torch.inverse(EVEC[notconverged,:(cFock+1),:(cFock+1)])[...,:-1,-1]
            F[notconverged] = torch.sum(FOCK[notconverged,:cFock,:,:]*coeff.unsqueeze(-1).unsqueeze(-1), dim=1)
            P[notconverged] = unpack( SP2(
                                          pack(F[notconverged], nHeavy[notconverged], nHydro[notconverged]),
                                          nOccMO[notconverged], sp2[1]
                                         ),
                                     nHeavy[notconverged], nHydro[notconverged], 4 * molsize)
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
            if k >= max_scf_iter: return P, notconverged
        else:
            return P, notconverged
    

