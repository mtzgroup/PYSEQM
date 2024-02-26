import torch
from typing import Dict
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .fermi_q import Fermi_Q
from .G_XL_LR import G
from .canon_dm_prt import Canon_DM_PRT


debug = False


#@torch.jit.script
def scf_forward3(M, w, gss, gpp, gsp, gp2, hsp, nHydro, nHeavy, nOccMO, 
                 nmol: int, molsize: int, maskd, mask, idxi, idxj, P, eps: float, 
                 xl_bomd_params: Dict, sp2=[False, 1e-4], max_scf_iter: int=200):
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
    
    while(1):
#        start_time = time.time()
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
#                end_time = time.time()
                print(COUNTER, SCF_err.cpu().numpy(), err.cpu().numpy(), torch.sum(notconverged).item())#, end_time - start_time)
            if COUNTER >= max_scf_iter: return P, notconverged
        else:
            return P, notconverged
