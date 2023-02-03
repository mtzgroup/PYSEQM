import torch
from itertools import chain


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ParameterKernelModel(torch.nn.Module):
    def __init__(self, Z_ref, desc_ref):
        super(ParameterKernelModel, self).__init__()
        c1 = isinstance(Z_ref, list) and torch.is_tensor(Z_ref[0])
        c2 = isinstance(desc_ref, list) and torch.is_tensor(desc_ref[0])
        if not (c1 and c2):
            msg  = "Atomic numbers and descriptors have to be provided "
            msg += "as list of tensors"
            raise ValueError(msg)
        Zflat = list(chain(*[z_i.tolist() for z_i in Z_ref]))
        self.elements_ref = sorted(set(Zflat))
        Zall = torch.tensor(Zflat)
        self.X_ref, self.idx_ref = {}, {}
        for elm in self.elements_ref:
            idx   = [torch.where(s==elm)[0] for s in Z_ref]
            X_elm = torch.cat([d[idx[i]] for i, d in enumerate(desc_ref)])
            self.X_ref[elm]   = X_elm
            self.idx_ref[elm] = torch.where(Zall==elm)[0]
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, Alpha, Z, desc, expK=1):
        Zflat = list(chain(*[z_i.tolist() for z_i in Z]))
        elements = sorted(set(Zflat))
        Zall = torch.tensor(Zflat)
        if any(elm not in self.elements_ref for elm in elements):
            raise ValueError("Elements in reference and call inconsistent!")
        y = torch.zeros((Alpha.shape[0], Zall.numel()))
        for elm in elements:
            ## get atomistic neighborhood descriptors for element
            idx_in   = [torch.where(s==elm)[0] for s in Z]
            X_in     = torch.cat([d[idx_in[i]] for i, d in enumerate(desc)])
            ## get atomistic kernel and map to total prediction
            K_base   = torch.matmul(self.X_ref[elm], X_in.T)
            K_elm    = torch.pow(K_base, expK)
            y_elm    = torch.matmul(Alpha[:,self.idx_ref[elm]], K_elm)
            ## map to total prediction
            idx      = torch.where(Zall==elm)[0]
            y[:,idx] = y_elm
        return y
        
    

