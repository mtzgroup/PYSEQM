#############################################################################
# Wrappers for SEQC calculations                                            #
#                                                                           #
# Current (Feb/20)                                                          #
# TODO: . clean up module, remove unnecessary wrappers                      #
#       . typing                                                            #
#       . improvement of GPU performance? (seems to be in PYSEQM internals) #
#############################################################################

import torch
from torch.autograd import grad as agrad
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
from seqm.basics import Energy, Parser
from seqm.seqm_functions.constants import Constants
from .pyseqm_helpers import prepare_array, Orderator, get_default_parameters
from .abstract_base_wrapper import AbstractWrapper
from .kernels import ParameterKernel
from .seqm_core_runners import SEQM_singlepoint_core, SEQM_multirun_core
from .kernel_core_runners import AMASE_singlepoint_core, AMASE_multirun_core


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device('cuda')
    sp2_def = [True, 1e-5]
else:
    device = torch.device('cpu')
    sp2_def = [False]

# default SEQC settings
default_settings = {
                    'method'            : 'AM1',
                    'scf_eps'           : 1.0e-6,
                    'scf_converger'     : [0,0.15],
                    'scf_backward'      : 2,
                    'scf_backward_eps'  : 1e-4,
                    'sp2'               : sp2_def,
                    'pair_outer_cutoff' : 1.0e10,
                    'Hf_flag'           : False,
                   }



class SEQM_singlepoint(torch.nn.Module):
    def __init__(self, seqm_settings={}, mode="full"):
        super(SEQM_singlepoint, self).__init__()
        self.settings = seqm_settings
        self.core_runner = SEQM_singlepoint_core(seqm_settings, mode=mode)
    
    def forward(self, p, species, coordinates, custom_params=[], mode="full",
                custom_reference=None):
        self.orderer = Orderator(species, coordinates)
        Z_padded = self.orderer.species
        Z, xyz = self.orderer.prepare_input()
        xyz.requires_grad_(True)
        nodummy = (Z_padded > 0)
        Z_plain = [s[nodummy[i]] for i, s in enumerate(Z_padded)]
        nAtoms = torch.count_nonzero(Z, dim=1)
        shifts = [0]+torch.cumsum(nAtoms, 0).tolist()[:-1]
        subsort = [torch.argsort(s, descending=True) for s in Z_plain]
        ragged_idx = [(s+shifts[i]).tolist() for i, s in enumerate(subsort)]
        p_sorting = torch.tensor(list(chain(*ragged_idx)))
        p_sorted = p[:,p_sorting]
        res = self.core_runner(p_sorted, Z, xyz, custom_params=custom_params,
                               custom_reference=custom_reference)
        #TODO: DOUBLE-CHECK IF THIS IS NEEDED!
#        F_resort = self.orderer.reorder(F)
#        res[2] = F_resort
        self.results = self.core_runner.results
#        self.results['forces'] = F_resort
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    
class SEQM_multirun(torch.nn.Module):
    def __init__(self, species, coordinates, custom_params=[], mode="full",
                custom_reference=None, seqm_settings=None):
        super(SEQM_multirun, self).__init__()
        self.orderer = Orderator(species, coordinates)
        Z_padded = self.orderer.species
        Z, xyz = self.orderer.prepare_input()
        xyz.requires_grad_(True)
        nodummy = (Z_padded > 0)
        Z_plain = [s[nodummy[i]] for i, s in enumerate(Z_padded)]
        nAtoms = torch.count_nonzero(Z, dim=1)
        shifts = [0]+torch.cumsum(nAtoms, 0).tolist()[:-1]
        subsort = [torch.argsort(s, descending=True) for s in Z_plain]
        ragged_idx = [(s+shifts[i]).tolist() for i, s in enumerate(subsort)]
        self.p_sorting = torch.tensor(list(chain(*ragged_idx)))
        self.core_runner = SEQM_multirun_core(Z, xyz,
                                        custom_params=custom_params,
                                        custom_reference=custom_reference,
                                        seqm_settings=seqm_settings)
    
    def forward(self, p):
        p_sorted = p[:,self.p_sorting]
        res = self.core_runner(p_sorted)
        #TODO: DOUBLE-CHECK IF THIS IS NEEDED!
#        F_resort = self.orderer.reorder(F)
#        res[2] = F_resort
        self.results = self.core_runner.results
#        self.results['forces'] = F_resort
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
    
    def forward(self, Alpha):
        res = self.core_runner(Alpha)
        self.results = self.core_runner.results
        return res
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    


#### NEW IMPLEMENTATIONS ####



class elementwiseSEQM_trainer(AbstractWrapper):
    """
    Concrete wrapper for SEQC calculations with elementwise parameters.
    
    Parameters at instantiation:
    ----------------------------
      . custom_params, list: names of custom parameters (to optimize)
      . mode, str: if 'full' learn total parameters, 'delta': learn Delta
      . seqm_settings, dict: settings for SEQC calculations
    """
    def __init__(self, custom_params=[], seqm_settings=None, mode="full",
                 use_custom_reference=False, loss_type="RSSperAtom", 
                 loss_args=(), loss_kwargs={}):
        super(elementwiseSEQM_trainer, self).__init__(custom_params=custom_params,
                                loss_type=loss_type, loss_args=loss_args,
                                loss_kwargs=loss_kwargs)
        self.core_runner = SEQM_singlepoint_core(seqm_settings, mode=mode,
                                custom_params=custom_params,
                                use_custom_reference=use_custom_reference)
    
    def forward(self, p_elm, species, coordinates, custom_reference=None):
        """
        Gather results of SEQC calculation with elementwise custom parameters.
        
        Parameters:
        -----------
          . p_elm, torch.Tensor: custom parameters for elements. Ordering:
                p_elm[i,j]: parameter custom_params[i] for element j, where
                elements are ordered in descending order.
          . species, list/torch.Tensor: atomic numbers ordered in descending order
          . coordinates, list/torch.Tensor: coordinates ordered accordingly
          . custom_reference, torch.Tensor: if in 'delta' mode:
            p = custom_reference + input, default: use standard parameters
        """
        # create maps for elementwise parameters (input) to actual parameters
        nondummy = (species > 0).reshape(-1)
        Zall = species.reshape(-1)[nondummy]
        real_elements = sorted(set([0]+Zall.tolist()), reverse=True)[:-1]
        elm_map = [real_elements.index(z) for z in Zall]
        p = torch.stack([p_elm[:,map_i] for map_i in elm_map]).T.to(device)
        species = species.to(device)
        coordinates = coordinates.to(device)
        coordinates.requires_grad_(True)
        res = self.core_runner(p, species, coordinates,
                               custom_reference=custom_reference)
        self.results = self.core_runner.results
        return res
        
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    
class SEQM_trainer(AbstractWrapper):
    """
    Concrete wrapper for SEQC calculations with atomwise parameters.
    
    Parameters at instantiation:
    ----------------------------
      . custom_params, list: names of custom parameters (to optimize)
      . mode, str: if 'full' learn total parameters, 'delta': learn Delta
      . seqm_settings, dict: settings for SEQC calculations
    """
    def __init__(self, custom_params=[], seqm_settings=None, mode="full",
                 use_custom_reference=False, loss_type="RSSperAtom", 
                 loss_args=(), loss_kwargs={}):
        super(SEQM_trainer, self).__init__(custom_params=custom_params,
                                loss_type=loss_type, loss_args=loss_args,
                                loss_kwargs=loss_kwargs)
        self.core_runner = SEQM_singlepoint_core(seqm_settings, mode=mode,
                                custom_params=custom_params,
                                use_custom_reference=use_custom_reference)
    
    def forward(self, p, species, coordinates, custom_reference=None):
        """
        Gather results of SEQC calculation with elementwise custom parameters.
        
        Parameters:
        -----------
          . p, torch.Tensor: custom parameters for atoms. Ordering:
                p[i,j]: parameter custom_params[i] for atom j, where
                atoms (in each molecule) are ordered in descending order.
          . species, list/torch.Tensor: atomic numbers in descending order
          . coordinates, list/torch.Tensor: coordinates ordered accordingly
          . custom_reference, torch.Tensor: if in 'delta' mode:
            p = custom_reference + input, default: use standard parameters
        """
#        species = species.to(device)
#        coordinates = coordinates.to(device)
        coordinates.requires_grad_(True)
        res = self.core_runner(p, species, coordinates,
                               custom_reference=custom_reference)
        self.results = self.core_runner.results
        return res
        
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    
class AMASE_trainer(AbstractWrapper):
    """
    Concrete loss module to optimize regression vector for SEQC with
    kernel-predicted parameters.
    """
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 custom_params=None, seqm_settings=None, mode="full",
                 use_custom_reference=False, expK=1, loss_type="RSSperAtom",
                 loss_args=(), loss_kwargs={}):
        super(AMASE_trainer, self).__init__(custom_params=custom_params,
                                loss_type=loss_type, loss_args=loss_args,
                                loss_kwargs=loss_kwargs)
        if isinstance(reference_Z, list):
            with torch.no_grad():
                reference_Z = pad_sequence(reference_Z, batch_first=True)
        if isinstance(reference_desc, list):
            with torch.no_grad():
                reference_desc = pad_sequence(reference_desc, batch_first=True)
        reference_Z = reference_Z.to(device)
        reference_desc = reference_desc.to(device)
        self.core_runner = AMASE_singlepoint_core(reference_Z, reference_desc,
                                 reference_coordinates=reference_coordinates,
                                 custom_params=custom_params,
                                 use_custom_reference=use_custom_reference,
                                 seqm_settings=seqm_settings, mode=mode)

    def forward(self, A, species, coordinates, desc, expK=1, custom_reference=None):
#        species = species.to(device)
#        coordinates = coordinates.to(device)
        coordinates.requires_grad_(True)
        res = self.core_runner(A, species, coordinates, desc, expK=expK,
                               custom_reference=custom_reference)
        self.results = self.core_runner.results
        return res

    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

