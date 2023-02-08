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
    def __init__(self, species, coordinates, custom_params=None,
                 seqm_settings=None):
        ## initialize parent module
        super(SEQM_Loss, self).__init__(species, coordinates,
                                        custom_params=custom_params)
        self.runner = SEQM_multirun_core(self.species, self.coordinates,
            custom_params=self.custom_params, seqm_settings=seqm_settings)

    def run_calculation(self, p):
        return self.runner(p)



class AMASE_Loss(AbstractLoss):
    def __init__(self, species, coordinates, desc, reference_Z,
        reference_desc, reference_coordinates=None, custom_params=None,
        seqm_settings=None, mode="full", custom_reference=None, expK=1):
        ## initialize parent module
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




