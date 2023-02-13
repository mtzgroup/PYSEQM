#############################################################################
# Wrappers for SEQC calculations                                            #
#                                                                           #
# Current (Feb/12)                                                          #
# TODO: . REFACTOR AMASE_trainer                                            #
#       . MOVE WRAPPERS TO AbstractWrapper DERIVATIVES (cf. AMASE_trainer)  #
#       . typing                                                            #
#       . double-check GPU support!                                         #
#############################################################################

import torch
from torch.autograd import grad as agrad
from itertools import chain
from seqm.basics import Energy, Parser
from seqm.seqm_functions.constants import Constants
from .pyseqm_helpers import prepare_array, Orderator, get_default_parameters
from .abstract_base_wrapper import AbstractWrapper
from .kernels import ParameterKernel
from .seqm_core_runners import SEQM_singlepoint_core, SEQM_multirun_core
from .kernel_core_runners import AMASE_singlepoint_core, AMASE_multirun_core


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



class SEQM_singlepoint(torch.nn.Module):
    def __init__(self, seqm_settings={}, mode="full"):
        super(SEQM_singlepoint, self).__init__()
        self.settings = seqm_settings
        self.core_runner = SEQM_singlepoint_core(seqm_settings, mode=mode)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
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
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
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
        
    
#TODO: wrap this around AMASE_singlepoint_core or similar
class AMASE_trainer(AbstractWrapper):
    """
    Concrete loss module to optimize regression vector for SEQC with
    kernel-predicted parameters.
    """
    def __init__(self, reference_Z, reference_desc, reference_coordinates=None,
                 custom_params=None, seqm_settings=None, mode="full",
                 custom_reference=None, expK=1, loss_type="RSSperAtom", 
                 loss_args=(), loss_kwargs={}):
        super(AMASE_trainer, self).__init__(custom_params=custom_params, 
                                loss_type=loss_type, loss_args=loss_args,
                                loss_kwargs=loss_kwargs)
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        Z_ref.requires_grad_(False)
        if callable(reference_desc): raise NotImplementedError
        # elements and indexing for reference structures
        nondummy = (Z_ref > 0).reshape(-1)
        Zall = Z_ref.reshape(-1)[nondummy]
        self.ref_elements = sorted(set(Zall.tolist()))
        self.ref_idx = [torch.where(Zall==elm)[0] for elm in self.ref_elements]
        
        # set up kernel matrix and settings
        self.kernel = ParameterKernel(Z_ref, reference_desc)
        self.settings = default_settings
        self.settings.update(seqm_settings)
        method = seqm_settings.get("method", "nomethoddefined")
        if method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        if param_dir == "nodirdefined":
            raise ValueError("`seqm_settings` has to include 'parameter_file_dir'")
        self.const = Constants()
        self.settings['learned'] = custom_params
        self.settings['eig'] = True
        if mode == "full":
            self.process_p = self.full_prediction
        elif mode == "delta":
            if custom_reference is None:
                self.process_p = self.default_delta
            else:
                self.process_p = self.custom_delta
        else:
            raise ValueError("Unknown mode '"+mode+"'.")
        self.results = {}

    def full_prediction(self, p, *args, **kwargs):
        return p

    def default_delta(self, p, species=None, *args, **kwargs):
        p0 = get_default_parameters(species, method=self.settings["method"],
                        parameter_dir=self.settings["parameter_file_dir"],
                        param_list=self.custom_params)
        p0.requires_grad_(False)
        return p + p0

    def custom_delta(self, p, custom_ref=None, *args, **kwargs):
        custom_ref.requires_grad_(False)
        return p + custom_ref
    
    def forward(self, A, species, coordinates, desc, expK=1, custom_reference=None):
        # elements and indexing for input structures
        nondummy = (species > 0).reshape(-1)
        Zall = species.reshape(-1)[nondummy]
        elements = sorted(set(Zall.tolist()))
        if any(elm not in self.ref_elements for elm in elements):
            msg  = "Some element(s) of requested systems not in reference "
            msg += "structures! Aborting."
            raise ValueError(msg)
        self.settings['elements'] = torch.tensor(sorted(set([0] + elements)))
        elm_idx = [torch.where(Zall==elm)[0] for elm in elements]
        # get HOMO and LUMO indices
        my_parser = Parser(self.settings)
        n_occ = my_parser(self.const, species, coordinates)[4]
        homo = (n_occ-1).unsqueeze(-1)
        lumo = n_occ.unsqueeze(-1)
        Ks = self.kernel.get_sorted_kernel(elements, species, desc, expK=expK)
        pred = torch.zeros((len(self.custom_params), Zall.numel()))
        Alpha_K = list(map(lambda i, K : torch.matmul(A[:,i], K),
                           self.ref_idx, Ks))
        for i, idx in enumerate(elm_idx): pred[:,idx] = Alpha_K[i]
        p = self.process_p(pred, species=species, custom_ref=custom_reference)
        learnedpar = {par:p[i] for i, par in enumerate(self.custom_params)}
        calc = Energy(self.settings)
        coordinates.requires_grad_(True)
        try:
            res = calc(self.const, coordinates, species, learnedpar,
                       all_terms=True)
        except RuntimeError:
            p.register_hook(lambda grad: grad * LFAIL)
            coordinates.register_hook(lambda grad: grad * LFAIL)
            return LFAIL, LFAIL, LFAIL*torch.ones_like(coordinates), LFAIL
        with torch.no_grad(): masking = torch.where(res[-1], LFAIL, 1.)
        Eat_fin = res[0] * masking
        Etot_fin = res[1] * masking
        F = -agrad(res[1].sum(), coordinates, create_graph=True)[0]
        coordinates.requires_grad_(False)
        F_fin = F * masking[...,None,None]
        ehomo = torch.gather(res[6], 1, homo).reshape(-1)
        elumo = torch.gather(res[6], 1, lumo).reshape(-1)
        gap_fin = (elumo - ehomo) * masking
        # update results dict
        self.results['atomization'] = Eat_fin
        self.results['energy'] = Etot_fin
        self.results['forces'] = F_fin
        self.results['gap'] = gap_fin
        return Eat_fin, Etot_fin, F_fin, gap_fin
        

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
        
    

