import torch
from itertools import chain
from .pyseqm_helpers import prepare_array, get_default_parameters, get_parameter_names
from .seqm_runners import SEQM_singlepoint, SEQM_multirun


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ParameterKernel(torch.nn.Module):
    def __init__(self, Z_ref, desc_ref):
        super(ParameterKernel, self).__init__()
        c1 = isinstance(Z_ref, list) and torch.is_tensor(Z_ref[0])
        c2 = isinstance(desc_ref, list) and torch.is_tensor(desc_ref[0])
        if not (c1 and c2):
            msg  = "Atomic numbers and descriptors have to be provided "
            msg += "as list of tensors!"
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
    
    def get_element_kernel(self, elm, idx, desc, expK=1):
        X_in     = torch.cat([d[idx[i]] for i, d in enumerate(desc)])
        K_base   = torch.matmul(self.X_ref[elm], X_in.T)
        K_elm    = torch.pow(K_base, expK)
        return K_elm
    
    def get_kernel_dict(self, Z, desc, expK=1):
        Zflat = list(chain(*[z_i.tolist() for z_i in Z]))
        elements = sorted(set(Zflat))
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
        Zflat = list(chain(*[z_i.tolist() for z_i in Z]))
        elements = sorted(set(Zflat))
        Zall = torch.tensor(Zflat)
        if any(elm not in self.elements_ref for elm in elements):
            raise ValueError("Elements in reference and call inconsistent!")
        K = self.get_kernel_dict(Z, desc, expK=expK)
        y = torch.zeros((Alpha.shape[0], Zall.numel()))
        Alpha_K = list(map(lambda elm : 
                           torch.matmul(Alpha[:,self.idx_ref[elm]], K[elm]),
                           elements))
        elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        for i, idx in enumerate(elm_idx): y[:,idx] = Alpha_K[i]
#        for elm in elements:
#            ## get atomistic neighborhood descriptors for element
#            idx_in   = [torch.where(s==elm)[0] for s in Z]
#            K_elm    = self.get_element_kernel(elm, idx_in, desc, expK=expK)
#            y_elm    = torch.matmul(Alpha[:,self.idx_ref[elm]], K_elm)
#            ## map to total prediction
#            idx      = torch.where(Zall==elm)[0]
#            y[:,idx] = y_elm
        return y
        
    
class AMASEQC_singlepoint(torch.nn.Module):
    #TODO: Allow for `reference_desc` to be callable
    #TODO: Typing, some refactoring, clean-up, docs
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 seqm_settings={}, parameters=None, mode="full", 
                 custom_reference=False):
        super(AMASEQC_singlepoint, self).__init__()
        self.param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        self.method = seqm_settings.get("method", "nomethoddefined")
        if self.method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        
        Zflat = list(chain(*[z_i.tolist() for z_i in reference_Z]))
        self.n_ref = len(Zflat)
        if callable(reference_desc): raise NotImplementedError
        self.kernel = ParameterKernel(reference_Z, reference_desc)
        if parameters is None:
            self.parameters = get_parameter_names(method)
        else:
            self.parameters = parameters
        if mode == "full":
            self.process_prediction = self.full_prediction
        elif mode == "delta":
            if custom_reference:
                self.process_prediction = self.delta_custom
            else:
                self.process_prediction = self.delta_default
        else:
            raise ValueError("Unknown mode '"+mode+"'.")
        self.runner = SEQM_singlepoint(seqm_settings)
        
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def full_prediction(self, par, **kwargs):
        return par
    
    def delta_default(self, par, species=None, reference_par=None):
        reference_par = get_default_parameters(species,
                            method=self.method, parameter_dir=self.param_dir,
                            param_list=self.parameters)
        return reference_par + par
    
    def custom_delta(self, par, reference_par=None, **kwargs):
        return reference_par + par
    
    def forward(self, Alpha, Z, positions, desc, reference_params=None):
        msg = "`Alpha` inconsistent with (`parameters`, number of references)"
        assert (Alpha.shape == (len(self.parameters),self.n_ref)), msg
        species = prepare_array(Z, "atomic numbers")
        coordinates = prepare_array(positions, "coordinates")
        if callable(desc): raise NotImplementedError
        pred = self.kernel(Alpha, Z, desc)
        p = self.process_prediction(pred, species=species, 
                                    reference_par=reference_params)
        result = self.runner(p, species, coordinates, 
                             custom_params=self.parameters)
        return result
        
    
class AMASEQC_multirun(torch.nn.Module):
    #TODO: Allow for `reference_desc` to be callable
    #TODO: Typing, refactor, clean-up, docs
    def __init__(self, Z, desc, coordinates, reference_Z, reference_desc, 
                 reference_coordinates=None, seqm_settings={}, mode="full", 
                 parameters=None, custom_reference=None, expK=1):
        super(AMASEQC_multirun, self).__init__()
        # check input
        self.param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        if self.param_dir == "nodirdefined":
            raise ValueError("`seqm_settings` has to include 'parameter_file_dir'")
        self.method = seqm_settings.get("method", "nomethoddefined")
        if self.method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        
        if callable(reference_desc): raise NotImplementedError
        if callable(desc): raise NotImplementedError
        if parameters is None:
            self.parameters = get_parameter_names(method)
        else:
            self.parameters = parameters
        
        # default parameters for method (as reference or as template)
        species = prepare_array(Z, "atomic numbers")
        p_def = get_default_parameters(species,
                            method=self.method, parameter_dir=self.param_dir,
                            param_list=self.parameters)
        # set p0 depending on mode (final p = prediction + p0)
        if mode == "full":
            self.p0 = torch.zeros_like(p_def)
        elif mode == "delta":
            if custom_reference is None:
                self.p0 = p_def.clone()
            else:
                self.p0 = custom_reference
        else:
            raise ValueError("Unknown mode '"+mode+"'.")
        self.p0.requires_grad_(False)
        
        # elements and indexing for input structures
        Zflat = list(chain(*[z_i.tolist() for z_i in Z]))
        elements = sorted(set(Zflat))
        Zall = torch.tensor(Zflat)
        self.elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        # elements and indexing for reference structures
        Zflat = list(chain(*[z_i.tolist() for z_i in reference_Z]))
        ref_elements = sorted(set(Zflat))
        if any(elm not in ref_elements for elm in elements):
            msg  = "Some element(s) of requrested systems not in reference "
            msg += "structures! Aborting."
            raise ValueError(msg)
        Zall = torch.tensor(Zflat)
        self.ref_idx = [torch.where(Zall==elm)[0] for elm in ref_elements]
        
        # build kernel matrix
        kernel = ParameterKernel(reference_Z, reference_desc)
        self.K = kernel.get_sorted_kernel(elements, Z, desc, expK=expK)
        # set up multirun calculator
        positions = prepare_array(coordinates, "coordinates")
        self.runner = SEQM_multirun(species, positions,
                                    custom_params=self.parameters,
                                    seqm_settings=seqm_settings)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, Alpha):
        pred = torch.zeros_like(self.p0)
        Alpha_K = list(map(lambda i, K : torch.matmul(Alpha[:,i], K), 
                           self.ref_idx, self.K))
        for i, idx in enumerate(self.elm_idx): pred[:,idx] = Alpha_K[i]
        p = pred + self.p0
        result = self.runner(p)
        return result
        
    

