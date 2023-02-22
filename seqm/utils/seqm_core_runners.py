#############################################################################
# Utilities for simplifying running SEQM calculations                       #
#  - SEQM_singlepoint: parameters, species, coordinates defined at runtime  #
#  - SEQM_multirun: parameters at runtime, systems fixed (offline train)    #
#                                                                           #
# Current (Feb/21):                                                         #
# TODO: . typing                                                            #
#       . custom backwards for (unlikely) RuntimeErrors in foward           #
#       . improvement of GPU performance? (seems to be in PYSEQM internals) #
#############################################################################

import torch
from torch.autograd import grad as agrad
from seqm.basics import Parser, Energy
from seqm.seqm_functions.constants import Constants
from .pyseqm_helpers import get_default_parameters


LFAIL = torch.tensor([1e6])

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


class SEQM_singlepoint_core(torch.nn.Module):
    """
    Core routine for running SEQC calculation for molecules
    (see seqm.utils.wrappers.SEQC_singlepoint for convenient wrapper)
    In singlepoint mode, SEQC parameters, atomic numbers, positions, custom
    parameter names, and custom reference parameters (`delta` mode, see below)
    are set at runtime.
    
    Parameters at instantiation:
      . seqm_settings, dict: settings for SEQC calculation
      . mode, str: if 'full', input parameters of call are SEQC parameters
                   if 'delta', SEQC parameters = input + reference params
                   default: 'full'
      . custom_params, list of str: names of custom parameters in p
            default: [] (i.e., use only standard parameters)
      . use_custom_reference, bool: whether to use custom reference parameters
                   for 'delta' mode (p = input + custom_reference)
    
    Parameters at call/forward:
      . p, torch.Tensor: SEQC parameters (or Delta parameters in 'delta' mode)
      . species, torch.Tensor: atomic numbers (sorted in descending order)
      . coordinates, torch.Tensor: atomic positions (ordered accordingly)
      . custom_reference, torch.Tensor: In 'delta' mode, SEQC parameters are
            given by `p` + `custom_reference`, default: None. If in 'delta' 
            mode and `custom_reference` is None: use standard parameters of
            method (as defined in `seqm_settings`) as reference params.
    
    Call/forward returns:
    ---------------------
      . Eat, torch.Tensor: atomization energy for each molecule
      . Etot, torch.Tensor: total energy for each molecule
      . F, torch.Tensor: atomic forces for each molecule
      . gap, torch.Tensor: HOMO-LUMO gap for each molecule
    
    Call/forward sets:
    ------------------
      . self.results, dict: dictionary containing Eat, Etot, F, gap
            (accessible through self.get_property(prop)
    
    """
    def __init__(self, seqm_settings={}, mode="full", custom_params=[],
                 use_custom_reference=False):
        super(SEQM_singlepoint_core, self).__init__()
        with torch.no_grad():
            self.settings = default_settings
            self.settings.update(seqm_settings)
            self.param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
            self.custom_params = custom_params
            self.method = seqm_settings.get("method", "nomethoddefined")
            if self.method == "nomethoddefined":
                raise ValueError("`seqm_settings` has to include 'method'")
            # set preprocessing of input -> SEQC parameters depending on `mode`
            param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
            if mode == "full":
                self.process_prediction = self.full_prediction
            elif mode == "delta":
                if use_custom_reference is False:
                    if param_dir == "nodirdefined":
                        msg  = "In 'delta' mode, `seqm_settings` has to include "
                        msg += "'parameter_file_dir'"
                        raise ValueError(msg)
                    self.process_prediction = self.default_delta
                else:
                    self.process_prediction = self.custom_delta
            else:
                raise ValueError("Unknown mode '"+mode+"'.")
            self.const = Constants().to(device)
            self.results = {}

    def full_prediction(self, par, **kwargs):
        return par

    def default_delta(self, par, species=None, **kwargs):
        p0 = get_default_parameters(species, method=self.settings["method"],
                        parameter_dir=self.settings["parameter_file_dir"],
                        param_list=self.custom_params).to(device)
        p0.requires_grad_(False)
        return par + p0

    def custom_delta(self, par, custom_ref=None, **kwargs):
        custom_ref.requires_grad_(False)
        return par + custom_ref
    
#    @staticmethod
    def forward(self, p, species, coordinates, custom_reference=None):
#    TODO: CUSTOM BACKWARD FOR WHEN CALCULTION FAILS?
#          IS p.register_hook(lambda grad: ...) working?
#    def forward(self, ctx, p):
        """ Run calculation. """
        elements = sorted(set([0] + species.reshape(-1).tolist()))
        p_in = self.process_prediction(p, species=species,
                                       reference_par=custom_reference)
        learnedpar = {par:p_in[i] for i, par in enumerate(self.custom_params)}
        self.settings['elements'] = torch.tensor(elements)
        self.settings['learned'] = self.custom_params
        self.settings['eig'] = True
        calc = Energy(self.settings).to(device)
#        try:
        res = calc(self.const, coordinates, species, learnedpar, 
                       all_terms=True)
#        except RuntimeError:
#            p.register_hook(lambda grad: grad * LFAIL)
#            coordinates.register_hook(lambda grad: grad * LFAIL)
#            return LFAIL, LFAIL, LFAIL*torch.ones_like(xyz), LFAIL
        masking = torch.where(res[-1], LFAIL, 1.)
        # atomization and total energy
        Eat_fin = res[0] * masking
        Etot_fin = res[1] * masking
        # forces
        F = -agrad(res[1].sum(), coordinates, create_graph=True)[0]
        F_fin = F * masking[...,None,None]
        # HOMO-LUMO gap
        my_parser = Parser(calc.seqm_parameters)
        n_occ = my_parser(self.const, species, coordinates)[4]
        homo = (n_occ-1).unsqueeze(-1)
        lumo = n_occ.unsqueeze(-1)
        ehomo = torch.gather(res[6], 1, homo).reshape(-1)
        elumo = torch.gather(res[6], 1, lumo).reshape(-1)
        gap_fin = (elumo - ehomo) * masking
        # update self.results dict
        self.results['atomization'] = Eat_fin
        self.results['energy'] = Etot_fin
        self.results['forces'] = F_fin
        self.results['gap'] = gap_fin
        return Eat_fin, Etot_fin, F_fin, gap_fin
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

class SEQM_multirun_core(torch.nn.Module):
    """
    Core routine for running multiple SEQC calculation for molecules
    (see seqm.utils.wrappers.SEQC_multirun for convenient wrapper)
    In multirun mode, only SEQC parameters (or Deltas) are set at runtime.
    This is mostly for parameter optimization.
    
    Parameters at instantiation:
      . seqm_settings, dict: settings for SEQC calculation
      . mode, str: if 'full', input parameters of call are SEQC parameters
                   if 'delta', SEQC parameters = input + reference params
            default: 'full'
      . species, torch.Tensor: atomic numbers (sorted in descending order)
      . coordinates, torch.Tensor: atomic positions (ordered accordingly)
      . custom_params, list of str: names of custom parameters in p
            default: [] (i.e., all standard parameters)
      . custom_reference, torch.Tensor: In 'delta' mode, SEQC parameters are
            given by `p` + `custom_reference`, default: None. If in 'delta' 
            mode and `custom_reference` is None: use standard parameters of
            method (as defined in `seqm_settings`) as reference params.
    
    Parameters at call/forward:
      . p, torch.Tensor: SEQC parameters (or Delta parameters in 'delta' mode)
    
    Call/forward returns:
    ---------------------
      . Eat, torch.Tensor: atomization energy for each molecule
      . Etot, torch.Tensor: total energy for each molecule
      . F, torch.Tensor: atomic forces for each molecule
      . gap, torch.Tensor: HOMO-LUMO gap for each molecule
    
    Call/forward sets:
    ------------------
      . self.results, dict: dictionary containing Eat, Etot, F, gap
            (accessible through self.get_property(prop)
    
    """
    def __init__(self, species, coordinates, custom_params=[], 
                 seqm_settings=None, mode="full", custom_reference=None):
        # initialize parent and gather attributes from input
        super(SEQM_multirun_core, self).__init__()
        settings = default_settings
        settings.update(seqm_settings)
        method = seqm_settings.get("method", "nomethoddefined")
        if method == "nomethoddefined":
            raise ValueError("`seqm_settings` has to include 'method'")
        param_dir = seqm_settings.get("parameter_file_dir", "nodirdefined")
        if param_dir == "nodirdefined":
            raise ValueError("`seqm_settings` has to include 'parameter_file_dir'")
        self.const = Constants()
        self.Z, self.xyz = species, coordinates
        self.xyz.requires_grad_(True)
        self.custom_par = custom_params
        elements = sorted(set([0] + self.Z.reshape(-1).tolist()))
        settings['elements'] = torch.tensor(elements)
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
        n_occ = my_parser(self.const, self.Z, self.xyz)[4]
        self.homo = (n_occ-1).unsqueeze(-1)
        self.lumo = n_occ.unsqueeze(-1)
        self.results = {}
    
#    @staticmethod
    def forward(self, p):
#    TODO: NEED CUSTOM BACKWARD FOR WHEN CALCULTION FAILS (RETURN NaN).
#          IS p.register_hook(lambda grad: grad * NaN) working?
#    def forward(self, ctx, p):
        """ Run calculation. """
        # preprocess input according to `mode`
        p_proc = p + self.p0
        learnedpar = {par:p_proc[i] for i, par in enumerate(self.custom_par)}
        try:
            res = self.calc(self.const, self.xyz, self.Z, learnedpar,
                            all_terms=True)
        except RuntimeError:
            p.register_hook(lambda grad: grad * LFAIL)
            self.xyz.register_hook(lambda grad: grad * LFAIL)
            return LFAIL, LFAIL, LFAIL*torch.ones_like(self.xyz), LFAIL
        masking = torch.where(res[-1], LFAIL, 1.)
        Eat_fin = res[0] * masking
        Etot_fin = res[1] * masking
        F = -agrad(res[1].sum(), self.xyz, create_graph=True)[0]
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
        
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

