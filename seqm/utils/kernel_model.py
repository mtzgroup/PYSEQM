############################################################################
# Utilities for running SEQM calculations with kernel-predicted parameters #
#  - ParameterKernel: get kernels and kernel-predicted parameters          #
#  - AMASE_singlepoint: Regression coeffs, atomic numbers, positions/      #
#                         descriptors defined at runtime                   #
#  - AMASE_multirun: Regression coefficients at runtime, systems fixed     #
#                                                                          #
# Curent (Feb/06): Basic implementation                                    #
# TODO:  . typing                                                          #
#        . refactor                                                        #
#        . ?add MD engine?                                                 #
#        . custom backwards for (unlikely) RuntimeErrors in foward         #
############################################################################

import torch
from itertools import chain
from .pyseqm_helpers import (prepare_array, Orderator,
    get_default_parameters, get_parameter_names)
from .seqm_runners import SEQM_singlepoint_core, SEQM_multirun_core


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ParameterKernel(torch.nn.Module):
    def __init__(self, Z_ref, desc_ref):
        super(ParameterKernel, self).__init__()
        if not torch.is_tensor(Z_ref):
            msg  = "Atomic numbers have to be provided as tensors!"
            raise ValueError(msg)
        if not isinstance(desc_ref, list) and torch.is_tensor(desc_ref[0]):
            msg = "Descriptors have to be provided as list of tensors!"
            raise ValueError(msg)
        nondummy = (Z_ref > 0).reshape(-1)
        Zall = Z_ref.reshape(-1)[nondummy]
        self.elements_ref = sorted(set(Zall.tolist()))
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
        nondummy = (Z > 0).reshape(-1)
        elements = sorted(set(Z.reshape(-1)[nondummy].tolist()))
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
        nondummy = (Z > 0).reshape(-1)
        Zall = Z.reshape(-1)[nondummy]
        elements = sorted(set(Zall.tolist()))
        if any(elm not in self.elements_ref for elm in elements):
            raise ValueError("Elements in reference and call inconsistent!")
        K = self.get_kernel_dict(Z, desc, expK=expK)
        y = torch.zeros((Alpha.shape[0], Zall.numel()))
        Alpha_K = list(map(lambda elm : 
                           torch.matmul(Alpha[:,self.idx_ref[elm]], K[elm]),
                           elements))
        elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        for i, idx in enumerate(elm_idx): y[:,idx] = Alpha_K[i]
        return y
        
    
class AMASE_singlepoint_core(torch.nn.Module):
    #TODO: Allow for `reference_desc` to be callable
    #TODO: Typing, some refactoring, clean-up, docs
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 seqm_settings={}, custom_params=None, mode="full", 
                 custom_reference=False):
        super(AMASE_singlepoint_core, self).__init__()
        self.param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        self.method = seqm_settings.get("method", "nomethoddefined")
        if self.method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        
        self.n_ref = torch.count_nonzero(reference_Z)
        if callable(reference_desc): raise NotImplementedError
        self.kernel = ParameterKernel(reference_Z, reference_desc)
        if custom_params is None:
            self.custom_params = get_parameter_names(method)
        else:
            self.custom_params = custom_params
        if mode == "full":
            self.process_prediction = self.full_prediction
        elif mode == "delta":
            if custom_reference:
                self.process_prediction = self.delta_custom
            else:
                self.process_prediction = self.delta_default
        else:
            raise ValueError("Unknown mode '"+mode+"'.")
        self.seqm_runner = SEQM_singlepoint_core(seqm_settings)
        self.results = {}
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def full_prediction(self, par, **kwargs):
        return par
    
    def delta_default(self, par, species=None, reference_par=None):
        reference_par = get_default_parameters(species,
                            method=self.method, parameter_dir=self.param_dir,
                            param_list=self.custom_params)
        return reference_par + par
    
    def custom_delta(self, par, reference_par=None, **kwargs):
        return reference_par + par
    
    def forward(self, Alpha, Z, positions, desc, reference_params=None, expK=1):
        msg = "`Alpha` inconsistent with (`custom_params`, number of references)"
        assert (Alpha.shape == (len(self.custom_params),self.n_ref)), msg
        if callable(desc): raise NotImplementedError
        pred = self.kernel(Alpha, Z, desc, expK=expK)
        p = self.process_prediction(pred, species=Z, 
                                    reference_par=reference_params)
        res = self.seqm_runner(p, Z, positions, custom_params=self.custom_params)
        self.results = self.seqm_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

class AMASE_singlepoint(torch.nn.Module):
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 seqm_settings={}, custom_params=None, mode="full",
                 custom_reference=False):
        super(AMASE_singlepoint, self).__init__()
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        self.core_runner = AMASE_singlepoint_core(Z_ref, reference_desc, 
                reference_coordinates=reference_coordinates, 
                seqm_settings=seqm_settings, custom_params=custom_params,
                mode=mode, custom_reference=custom_reference)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, Alpha, Z, positions, desc, reference_params=None, expK=1):
        orderer = Orderator(Z, positions)
        species, xyz = orderer.prepare_input()
        xyz.requires_grad_(True)
        res = self.core_runner(Alpha, species, xyz, desc, expK=expK,
                               reference_params=reference_params)
        self.results = self.core_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

class AMASE_multirun_core(torch.nn.Module):
    #TODO: Allow for `reference_desc` to be callable
    #TODO: Typing, refactor, clean-up, docs
    def __init__(self, Z, desc, coordinates, reference_Z, reference_desc, 
                 reference_coordinates=None, seqm_settings={}, mode="full", 
                 custom_params=None, custom_reference=None, expK=1):
        super(AMASE_multirun_core, self).__init__()
        # check input
        if not torch.is_tensor(Z):
            msg = "In core routine, atomic numbers have to be a tensor!"
            raise ValueError(msg)
        if not torch.is_tensor(coordinates):
            msg = "In core routine, coordinates have to be a tensor!"
            raise ValueError(msg)
        self.param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        if self.param_dir == "nodirdefined":
            raise ValueError("`seqm_settings` has to include 'parameter_file_dir'")
        self.method = seqm_settings.get("method", "nomethoddefined")
        if self.method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        
        if callable(reference_desc): raise NotImplementedError
        if callable(desc): raise NotImplementedError
        if custom_params is None:
            self.custom_params = get_parameter_names(method)
        else:
            self.custom_params = custom_params
        
        # default parameters for method (as reference or as template)
        p_def = get_default_parameters(Z, method=self.method, 
                    parameter_dir=self.param_dir, param_list=self.custom_params)
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
        nondummy = (Z > 0).reshape(-1)
        Zall = Z.reshape(-1)[nondummy]
        elements = sorted(set(Zall.tolist()))
        self.elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        # elements and indexing for reference structures
        nondummy = (reference_Z > 0).reshape(-1)
        Zall = reference_Z.reshape(-1)[nondummy]
        ref_elements = sorted(set(Zall.tolist()))
        if any(elm not in ref_elements for elm in elements):
            msg  = "Some element(s) of requrested systems not in reference "
            msg += "structures! Aborting."
            raise ValueError(msg)
        self.ref_idx = [torch.where(Zall==elm)[0] for elm in ref_elements]
        
        # build kernel matrix
        kernel = ParameterKernel(reference_Z, reference_desc)
        self.K = kernel.get_sorted_kernel(elements, Z, desc, expK=expK)
        # set up multirun calculator
        self.seqm_runner = SEQM_multirun_core(Z, coordinates,
                                      custom_params=self.custom_params,
                                      seqm_settings=seqm_settings)
        self.results = {}
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, Alpha):
        pred = torch.zeros_like(self.p0)
        Alpha_K = list(map(lambda i, K : torch.matmul(Alpha[:,i], K), 
                           self.ref_idx, self.K))
        for i, idx in enumerate(self.elm_idx): pred[:,idx] = Alpha_K[i]
        p = pred + self.p0
        res = self.seqm_runner(p)
        self.results = self.seqm_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    
class AMASE_multirun(torch.nn.Module):
    def __init__(self, Z, desc, coordinates, reference_Z, reference_desc,
                 reference_coordinates=None, seqm_settings={}, mode="full",
                 custom_params=None, custom_reference=None, expK=1):
        super(AMASE_multirun, self).__init__()
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        orderer = Orderator(Z, coordinates)
        species, xyz = orderer.prepare_input()
        xyz.requires_grad_(True)
        self.core_runner = AMASE_multirun_core(species, desc, xyz, Z_ref, 
                reference_desc, reference_coordinates=reference_coordinates,
                seqm_settings=seqm_settings, mode=mode, expK=expK,
                custom_params=custom_params, custom_reference=custom_reference)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, Alpha):
        res = self.core_runner(Alpha)
        self.results = self.core_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

