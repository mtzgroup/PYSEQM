############################################################################
# Concrete loss modules for SEQC with custom/adaptive parameters           #
#  - SEQM_Loss: loss for regular SEQC calculation w.r.t. parameters        #
#  - AMASE_Loss: loss for SEQC with kernel-predicted parameters            #
#                                                                          #
# Current (Feb/07)                                                         #
# TODO:  . typing                                                          #
#        . double-check GPU support!                                       #
############################################################################

import torch
from .pyseqm_helpers import prepare_array
from .abstract_base_loss import AbstractLoss
from .seqm_core_runners import SEQM_multirun_core
from .kernel_core_runners import AMASE_multirun_core


class SEQM_Loss(AbstractLoss):
    """ Concrete loss module to optimize parameters in SEQC."""
    def __init__(self, species, coordinates, custom_params=[], mode="full",
                 seqm_settings={}, custom_reference=None):
        super(SEQM_Loss, self).__init__(species, coordinates,
                                        custom_params=custom_params)
        self.runner = SEQM_multirun_core(self.species, self.coordinates,
            custom_params=self.custom_params, seqm_settings=seqm_settings,
            mode=mode, custom_reference=custom_reference)
    
    def run_calculation(self, p):
        return self.runner(p)
        
    

class elementwiseSEQM_Loss(AbstractLoss):
    """
    Concrete loss module to optimize elementwise parameters in SEQC.
    
    Parameters at instantiation:
    ----------------------------
      . species, list/torch.Tensor: atomic numbers ordered in descending order
      . coordinates, list/torch.Tensor: coordinates ordered accordingly
      . custom_params, list: names of custom parameters (to optimize)
      . mode, str: if 'full' learn total parameters, 'delta': learn Delta
      . seqm_settings, dict: settings for SEQC calculations
      . custom_reference, torch.Tensor: if in 'delta' mode:
            p = custom_reference + input, default: use standard parameters
    
    """
    def __init__(self, species, coordinates, custom_params=[], mode="full",
                 seqm_settings={}, custom_reference=None):
        super(SEQM_Loss, self).__init__(species, coordinates,
                                        custom_params=custom_params)
        # create maps for elementwise parameters (input) to actual parameters
        nondummy = (Z > 0).reshape(-1)
        self.Zall = Z.reshape(-1)[nondummy]
        real_elements = sorted(set([0]+Zall.tolist()), reverse=True)[:-1]
        self.elm_map = [real_elements.index(z) for z in Zall]
        self.runner = SEQM_multirun_core(self.species, self.coordinates,
            custom_params=self.custom_params, seqm_settings=seqm_settings,
            mode=mode, custom_reference=custom_reference)
    
    def run_calculation(self, p_elm):
        """
        Gather results of SEQC calculation with elementwise custom parameters.
        
        Parameters:
        -----------
          . p_elm, torch.Tensor: custom parameters for elements. Ordering:
                p_elm[i,j]: parameter custom_params[i] for element j, where
                elements are ordered in descending order.
        """
        p = torch.stack([p_elm[:,map_i] for map_i in elm_map]).T
        return self.runner(p)
        
    

class AMASE_Loss(AbstractLoss):
    """
    Concrete loss module to optimize regression vector for SEQC with
    kernel-predicted parameters.
    """
    def __init__(self, species, coordinates, desc, reference_Z,
        reference_desc, reference_coordinates=None, custom_params=None,
        seqm_settings=None, mode="full", custom_reference=None, expK=1):
        super(AMASE_Loss, self).__init__(species, coordinates,
                                         custom_params=custom_params)
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        self.runner = AMASE_multirun_core(self.species, desc,
                self.coordinates, Z_ref, reference_desc,
                reference_coordinates=reference_coordinates,
                seqm_settings=seqm_settings, mode=mode,
                custom_params=custom_params, expK=expK,
                custom_reference=custom_reference)

    def run_calculation(self, A):
        return self.runner(A)




