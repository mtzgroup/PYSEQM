############################################################################
# Concrete loss modules for SEQC with custom/adaptive parameters           #
#  - SEQM_Loss: loss for regular SEQC calculation w.r.t. parameters        #
#  - AMASE_Loss: loss for SEQC with kernel-predicted parameters            #
#                                                                          #
# Current (Feb/09)                                                         #
# TODO:  . typing                                                          #
#        . double-check GPU support!                                       #
#        . enable select-atoms/elements-only mode                          #
############################################################################

import torch
from torch.autograd import grad as agrad
from seqm.basics import Parser, Energy
from seqm.seqm_functions.constants import Constants
from .pyseqm_helpers import prepare_array, get_default_parameters
from .abstract_base_loss import AbstractLoss
from .seqm_core_runners import SEQM_multirun_core
#from .kernel_core_runners import AMASE_multirun_core
from .kernels import ParameterKernel


LFAIL = torch.tensor(torch.inf, requires_grad=True)

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



class SEQM_Loss(AbstractLoss):
    """ Concrete loss module to optimize parameters in SEQC."""
    def __init__(self, species, coordinates, custom_params=[], mode="full",
            seqm_settings={}, custom_reference=None, loss_type="RSSperAtom",
            loss_args=(), loss_kwargs={}):
        super(SEQM_Loss, self).__init__(species, coordinates,
                        custom_params=custom_params, loss_type=loss_type,
                        loss_args=loss_args, loss_kwargs=loss_kwargs)
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
            seqm_settings={}, custom_reference=None, loss_type="RSSperAtom",
            loss_args=(), loss_kwargs={}):
        super(SEQM_Loss, self).__init__(species, coordinates,
                        custom_params=custom_params, loss_type=loss_type,
                        loss_args=loss_args, loss_kwargs=loss_kwargs)
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
            seqm_settings=None, mode="full", custom_reference=None, expK=1,
            loss_type="RSSperAtom", loss_args=(), loss_kwargs={}):
        super(AMASE_Loss, self).__init__(species, coordinates,
                        custom_params=custom_params, loss_type=loss_type,
                        loss_args=loss_args, loss_kwargs=loss_kwargs)
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        if callable(reference_desc): raise NotImplementedError
        if callable(desc): raise NotImplementedError
        # elements and indexing for input structures
        nondummy = (self.species > 0).reshape(-1)
        Zall = self.species.reshape(-1)[nondummy]
        elements = sorted(set(Zall.tolist()))
        self.elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        self.p_shape = (len(custom_params), Zall.numel())
        # elements and indexing for reference structures
        nondummy = (Z_ref > 0).reshape(-1)
        Zall = Z_ref.reshape(-1)[nondummy]
        ref_elements = sorted(set(Zall.tolist()))
        if any(elm not in ref_elements for elm in elements):
            msg  = "Some element(s) of requested systems not in reference "
            msg += "structures! Aborting."
            raise ValueError(msg)
        self.ref_idx = [torch.where(Zall==elm)[0] for elm in ref_elements]
        
        # build kernel matrix
        kernel = ParameterKernel(Z_ref, reference_desc)
        self.K = kernel.get_sorted_kernel(elements, self.species, desc, expK=expK)
        # set up multirun calculator
        settings = default_settings
        settings.update(seqm_settings)
        method = seqm_settings.get("method", "nomethoddefined")
        if method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        if param_dir == "nodirdefined":
            raise ValueError("`seqm_settings` has to include 'parameter_file_dir'")
        self.const = Constants()
        self.custom_par = custom_params
        settings['elements'] = torch.tensor(sorted(set([0] + elements)))
        settings['learned'] = custom_params
        settings['eig'] = True
        # default parameters for method (as reference or as template)
        p_def = get_default_parameters(species, method=method,
                    parameter_dir=param_dir, param_list=custom_params)
        # set `self.p0` depending on `mode` (final `p` = input + `self.p0`)
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
        self.calc = Energy(settings)
        # get HOMO and LUMO indices
        my_parser = Parser(self.calc.seqm_parameters)
        n_occ = my_parser(self.const, self.species, self.coordinates)[4]
        self.homo = (n_occ-1).unsqueeze(-1)
        self.lumo = n_occ.unsqueeze(-1)
        self.results = {}
        
    
    def run_calculation(self, A):
        pred = torch.zeros(self.p_shape)
        Alpha_K = list(map(lambda i, K : torch.matmul(A[:,i], K),
                           self.ref_idx, self.K))
        for i, idx in enumerate(self.elm_idx): pred[:,idx] = Alpha_K[i]
        p = pred + self.p0
        learnedpar = {par:p[i] for i, par in enumerate(self.custom_par)}
        try:
            res = self.calc(self.const, self.coordinates, self.species,
                            learnedpar, all_terms=True)
        except RuntimeError:
            p.register_hook(lambda grad: grad * LFAIL)
            self.coordinates.register_hook(lambda grad: grad * LFAIL)
            return LFAIL, LFAIL, LFAIL*torch.ones_like(self.coordinates), LFAIL
        masking = torch.where(res[-1], LFAIL, 1.)
        Eat_fin = res[0] * masking
        Etot_fin = res[1] * masking
        F = -agrad(res[1].sum(), self.coordinates, create_graph=True)[0]
        F_fin = F * masking[...,None,None]
        ehomo = torch.gather(res[6], 1, self.homo).reshape(-1)
        elumo = torch.gather(res[6], 1, self.lumo).reshape(-1)
        gap_fin = (elumo - ehomo) * masking
        # update results dict
        self.results['atomization'] = Eat_fin
        self.results['energy'] = Etot_fin
        self.results['forces'] = F_fin
        self.results['gap'] = gap_fin
        return Eat_fin, Etot_fin, F_fin, gap_fin
        
    

