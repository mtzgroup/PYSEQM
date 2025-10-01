import torch
from typing import Union


class SimpleDIIS:
    def __init__(self, n_mol, nspin, n, k_max=8, rank_compression: Union[float, bool] = False,
                 device=None, dtype=None, seed=0, regularization=1e-8):
        self.k_max = k_max
        Ik = torch.eye(k_max, dtype=dtype, device=device).unsqueeze(0)
        self.reg_mat = regularization * Ik.expand(n_mol,-1,-1)
        self.rhs = torch.zeros((n_mol, k_max + 1, 1), dtype=dtype, device=device)
        self.rhs[:, -1, 0] = 1.0
        self.n_zeros = torch.zeros((n_mol, 1, 1), dtype=dtype, device=device)
        self.F_hist = torch.zeros((n_mol, k_max, nspin, n, n), dtype=dtype, device=device)
        self.in_buffer = torch.zeros((n_mol, k_max), dtype=dtype, device=device)
        self.pos, Ncomp = 0, nspin * n * n
        if rank_compression:
            if not isinstance(rank_compression, float):
                msg  = "Invalid input for `rank_compression`. Expected one of float or False,"
                msg += " but got " + str(type(rank_compression))
                raise ValueError(msg)
            elif (rank_compression > 1.0) or (rank_compression <= 0.01):
                msg  = "Invalid input for `rank_compression`. Expected float between 0.01 and 1 "
                msg += "or False, but got " + str(rank_compression)
                raise ValueError(msg)
            ## set up random projection
            torch.manual_seed(seed)
            self.FPPF_rank = int(rank_compression * Ncomp)
            V = torch.randn((self.FPPF_rank, Ncomp), dtype=dtype, device=device)
            Q, _ = torch.linalg.qr(V.T)
            self.proj = Q[:, :self.FPPF_rank].clone()
            self.proj.requires_grad_(False)
            self.prepare_FPPF = self._compress_FPPF
        else:
            self.FPPF_rank = Ncomp
            self.prepare_FPPF = self._reshape_FPPF
        self.comm_hist = torch.zeros((n_mol, k_max, self.FPPF_rank), dtype=dtype, device=device)

    def append(self, F, FPPF):
        """
        Append current F and commutator, FP - PF, into circular buffer.
        F, FPPF: (n_mol, nspin, n, n)
        """
        ## handle RHF Fock matrix
        if F.dim() == 3: F = F.unsqueeze(1)
        n_mol, nspin, n, _ = F.shape
        self.F_hist[:, self.pos] = F
        self.comm_hist[:, self.pos] = self.prepare_FPPF(FPPF, n_mol, nspin, n)
        self.in_buffer[:, self.pos] = 1
        ## update position in buffer
        self.pos = (self.pos + 1) % self.k_max

    def _compress_FPPF(self, FPPF, n_mol, nspin, n):
        """ compress FPPF using random projection """
        err_flat = FPPF.reshape(n_mol, nspin * n * n)
        err_comp = torch.einsum("...ik,...kj->ij", err_flat, self.proj)
        return err_comp.reshape(n_mol, self.FPPF_rank)
        
    def _reshape_FPPF(self, FPPF, n_mol, nspin, n):
        return FPPF.reshape(n_mol, self.FPPF_rank)
    
    def extrapolate(self):
        """ Do the DIIS. """
        n_mol, k, _, _, _ = self.F_hist.shape
        B = torch.einsum("...ik,...jk->ij", self.comm_hist, self.comm_hist)
        mask_col = self.in_buffer.unsqueeze(2)
        mask2 = mask_col * self.in_buffer.unsqueeze(1)
        B = B * mask2 + self.reg_mat
        top = torch.cat([B, mask_col], dim=2)
        bottom = torch.cat([mask_col.transpose(1,2), self.n_zeros], dim=2)
        A = torch.cat([top, bottom], dim=1)
        coeffs, _ = torch.linalg.solve_ex(A, self.rhs)
        c = coeffs.squeeze(-1)[:, :k]
        F_extrap = (c.view(n_mol,k,1,1,1) * self.F_hist).sum(dim=1)
        return F_extrap.squeeze(1)
    

