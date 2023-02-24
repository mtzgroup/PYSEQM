import torch


#TODO: abstract kernel construction (allow multiple kernels)
class ParameterKernel(torch.nn.Module):
    def __init__(self, Z_ref, desc_ref):
        super(ParameterKernel, self).__init__()
        if not torch.is_tensor(Z_ref):
            msg  = "Atomic numbers have to be provided as tensors!"
            raise ValueError(msg)
        with torch.no_grad():
            nondummy = (Z_ref > 0).reshape(-1)
            Zall = Z_ref.reshape(-1)[nondummy]
            self.elements_ref = sorted(set(Zall.tolist()))
            self.X_ref, self.idx_ref = {}, {}
            for elm in self.elements_ref:
                idx   = [torch.where(s==elm)[0] for s in Z_ref]
                X_elm = torch.cat([d[idx[i]] for i, d in enumerate(desc_ref)])
                X_elm.requires_grad_(False)
                self.X_ref[elm]   = X_elm
                self.idx_ref[elm] = torch.where(Zall==elm)[0]
    
    def get_element_kernel(self, elm, idx, desc, expK=1):
        X_in     = torch.cat([d[idx[i]] for i, d in enumerate(desc)])
        K_base   = torch.matmul(self.X_ref[elm], X_in.T)
        K_elm    = torch.pow(K_base, expK)
        return K_elm
    
    def get_kernel_dict(self, elements, Z, desc, expK=1):
        K_dict = {}
        for elm in elements:
            ## get atomistic neighborhood descriptors for element
            idx_in = [torch.where(s==elm)[0] for s in Z]
            K_dict[elm] = self.get_element_kernel(elm, idx_in, desc, expK=expK)
        return K_dict
    
    def get_sorted_kernel(self, elements, Z, desc, expK=1):
        K = [self.get_element_kernel(elm, [torch.where(s==elm)[0] for s in Z],
                                     desc, expK=expK) for elm in elements]
        return K
    
    def forward(self, Alpha, Z, desc, expK=1):
        with torch.no_grad():
            nondummy = (Z > 0).reshape(-1)
            Zall = Z.reshape(-1)[nondummy]
            elements = sorted(set(Zall.tolist()))
            if any(elm not in self.elements_ref for elm in elements):
                raise ValueError("Elements in reference and call inconsistent!")
            K = self.get_kernel_dict(elements, Z, desc, expK=expK)
        y = torch.zeros((Alpha.shape[0], Zall.numel())).to(Alpha.device)
        Alpha_K = list(map(lambda elm :
                           torch.matmul(Alpha[:,self.idx_ref[elm]], K[elm].to(Alpha.device)),
                           elements))
        elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        for i, idx in enumerate(elm_idx): y[:,idx] = Alpha_K[i]
        return y
        
    

