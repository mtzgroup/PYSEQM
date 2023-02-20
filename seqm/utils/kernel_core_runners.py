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
#        . ?add MD/geometry optimization engine?                           #
#        . custom backwards for (unlikely) RuntimeErrors in forward        #
############################################################################

import torch
from .kernels import ParameterKernel
from .seqm_core_runners import SEQM_singlepoint_core, SEQM_multirun_core



class AMASE_singlepoint_core(torch.nn.Module):
    #TODO: Allow for `reference_desc` to be callable
    #TODO: Typing, some refactoring, clean-up, docs
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 seqm_settings={}, mode="full", custom_params=[],
                 use_custom_reference=False):
        super(AMASE_singlepoint_core, self).__init__()
        self.n_ref = torch.count_nonzero(reference_Z)
        if callable(reference_desc): raise NotImplementedError
        self.kernel = ParameterKernel(reference_Z, reference_desc)
        self.seqm_runner = SEQM_singlepoint_core(seqm_settings, mode=mode,
                                custom_params=custom_params,
                                use_custom_reference=use_custom_reference)
        self.results = {}
    
    def forward(self, Alpha, Z, positions, desc, custom_reference=None, expK=1):
        pred = self.kernel(Alpha, Z, desc, expK=expK)
        res = self.seqm_runner(pred, Z, positions, 
                               custom_reference=custom_reference)
        self.results = self.seqm_runner.results
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
                 custom_params=[], custom_reference=None, expK=1):
        super(AMASE_multirun_core, self).__init__()
        # check input
        if not torch.is_tensor(Z):
            msg = "In core routine, atomic numbers have to be a tensor!"
            raise ValueError(msg)
        if not torch.is_tensor(coordinates):
            msg = "In core routine, coordinates have to be a tensor!"
            raise ValueError(msg)
        if callable(reference_desc): raise NotImplementedError
        if callable(desc): raise NotImplementedError
        # elements and indexing for input structures
        nondummy = (Z > 0).reshape(-1)
        Zall = Z.reshape(-1)[nondummy]
        elements = sorted(set(Zall.tolist()))
        self.elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        self.p_shape = (len(custom_params), Zall.numel())
        # elements and indexing for reference structures
        nondummy = (reference_Z > 0).reshape(-1)
        Zall = reference_Z.reshape(-1)[nondummy]
        ref_elements = sorted(set(Zall.tolist()))
        if any(elm not in ref_elements for elm in elements):
            msg  = "Some element(s) of requested systems not in reference "
            msg += "structures! Aborting."
            raise ValueError(msg)
        self.ref_idx = [torch.where(Zall==elm)[0] for elm in ref_elements]
        
        # build kernel matrix
        kernel = ParameterKernel(reference_Z, reference_desc)
        self.K = kernel.get_sorted_kernel(elements, Z, desc, expK=expK)
        # set up multirun calculator
        self.seqm_runner = SEQM_multirun_core(Z, coordinates, mode=mode,
                                      custom_params=custom_params,
                                      custom_reference=custom_reference,
                                      seqm_settings=seqm_settings)
        self.results = {}
        del(nondummy,Zall,desc,reference_desc)
    
    def forward(self, Alpha):
        pred = torch.zeros(self.p_shape)
        Alpha_K = list(map(lambda i, K : torch.matmul(Alpha[:,i], K), 
                           self.ref_idx, self.K))
        for i, idx in enumerate(self.elm_idx): pred[:,idx] = Alpha_K[i]
        res = self.seqm_runner(pred)
        self.results = self.seqm_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

