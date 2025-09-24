import torch


class DIIS:
    def __init__(self, n_mol, nspin, n, k_max=8, compress_rank=None,
                 device=None, dtype=None, seed=0):
        self.k_max = k_max
        self.F_hist = torch.zeros((n_mol, k_max, nspin, n, n),
                                  device=device, dtype=dtype)
        self.err_hist = torch.zeros_like(self.F_hist)
        self.filled = torch.zeros((n_mol, k_max), dtype=torch.bool, device=device)
        self.hist_lens = torch.zeros((n_mol,), dtype=torch.long, device=device)
        self.pos = 0
        self.compress_rank = None if compress_rank is None else int(compress_rank)
        Ncomp = nspin * n * n
        if self.compress_rank is not None:
            torch.manual_seed(seed)
            V = torch.randn((self.compress_rank, Ncomp), device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(V.T)
            self.proj = Q[:, :self.compress_rank].clone().to(device=device, dtype=dtype)
            self.proj.requires_grad_(False)
        else:
            self.proj = None

    def append(self, F, err):
        """
        Append current F and error (= FP - PF) into circular buffer.
        F, err: (n_mol, nspin, n, n)
        """
        ## handle RHF Fock matrix
        if F.dim() == 3: F = F.unsqueeze(1)
        self.F_hist[:, self.pos] = F
        self.err_hist[:, self.pos] = err
        self.filled[:, self.pos] = True
        self.hist_lens = torch.clamp(self.hist_lens + 1, max=self.k_max)
        self.pos = (self.pos + 1) % self.k_max

    def _compress_errors(self):
        n_mol, k, nspin, n, _ = self.err_hist.shape
        if self.proj is None:
            return self.err_hist.reshape(n_mol, k, nspin * n * n)
        E2 = self.err_hist.reshape(n_mol * k, nspin * n * n)
        C = E2 @ self.proj
        return C.reshape(n_mol, k, self.compress_rank)

    def extrapolate(self, reg=1e-8):
        n_mol, k, nspin, n, _ = self.F_hist.shape
        dtype, device = self.F_hist.dtype, self.F_hist.device
        Ecomp = self._compress_errors()
        B = Ecomp @ Ecomp.transpose(1,2)
        valid_mask = self.filled.to(dtype=dtype)
        mask2 = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        B = B * mask2
        eye_k = torch.eye(k, device=device, dtype=dtype).unsqueeze(0).expand(n_mol, -1, -1)
        B = B + reg * eye_k
        ones_col = valid_mask.unsqueeze(2)
        top = torch.cat([B, ones_col], dim=2)
        bottom = torch.cat([ones_col.transpose(1, 2), torch.zeros((n_mol, 1, 1),
                           device=device, dtype=dtype)], dim=2)
        A = torch.cat([top, bottom], dim=1)
        rhs = torch.zeros((n_mol, k + 1), device=device, dtype=dtype)
        rhs[:, -1] = 1.0
        coeffs = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        c = coeffs[:, :k]
        F_extrap = (c.view(n_mol, k, 1, 1, 1) * self.F_hist).sum(dim=1)
        short_mask = self.hist_lens < 2
        if short_mask.any():
            last_idx = (self.pos - 1) % self.k_max
            last_F = self.F_hist[:, last_idx]
            sm = short_mask.view(n_mol, 1, 1, 1)
            F_extrap = torch.where(sm, last_F, F_extrap)
        return F_extrap.squeeze(1)
    

