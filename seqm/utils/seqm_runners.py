############################################################################
# Utilities for running SEQM calculations                                  #
#  - SEQM_singlepoint: parameters, species, coordinates defined at runtime #
#  - SEQM_multirun: parameters at runtime, systems fixed (for training)    #
#                                                                          #
# Curent (Feb/06): Basic implementation                                    #
# TODO:  . typing                                                          #
#        . refactor                                                        #
#        . ?add MD engine?                                                 #
#        . custom backwards for (unlikely) RuntimeErrors in foward         #
############################################################################

import torch
from itertools import chain
from torch.autograd import grad as agrad
from seqm.basics import Parser, Energy
from seqm.seqm_functions.constants import Constants
from .pyseqm_helpers import Orderator


LFAIL = torch.tensor(torch.inf)

torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device('cuda')
    sp2_def = [True, 1e-5]
else:
    device = torch.device('cpu')
    sp2_def = [False]


default_settings = {
                    'method'            : 'AM1',
                    'scf_eps'           : 1.0e-6,
                    'scf_converger'     : [0,0.15],
                    'scf_backward'      : 2,
                    'sp2'               : sp2_def,
                    'pair_outer_cutoff' : 1.0e10,
                    'Hf_flag'           : False,
                   }


class SEQM_singlepoint_core(torch.nn.Module):
    def __init__(self, seqm_settings=None):
        super(SEQM_singlepoint_core, self).__init__()
        self.settings = default_settings
        self.settings.update(seqm_settings)
        self.const = Constants()
        self.results = {}
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__

#    @staticmethod
    def forward(self, p, species, coordinates, custom_params=[]):
#    TODO: NEED CUSTOM BACKWARD FOR WHEN CALCULTION FAILS (RETURN NaN).
#          IS p.register_hook(lambda grad: grad * NaN) working?
#    def forward(self, ctx, p):
        """ Run calculation. """
        elements = sorted(set([0] + species.reshape(-1).tolist()))
        learnedpar = {par:p[i] for i, par in enumerate(custom_params)}
        self.settings['elements'] = torch.tensor(elements)
        self.settings['learned'] = custom_params
        self.settings['eig'] = True
        calc = Energy(self.settings)
        try:
            res = calc(self.const, coordinates, species, learnedpar, 
                       all_terms=True)
        except RuntimeError:
            p.register_hook(lambda grad: grad * torch.nan)
            coordinates.register_hook(lambda grad: grad * torch.nan)
            return {'atomization':torch.nan, 'energy':torch.nan, 
                    'forces':torch.nan*torch.ones_like(xyz), 'gap':torch.nan}
        masking = (~res[-1]).float()
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
        self.results['atomization'] = Eat_fin
        self.results['energy'] = Etot_fin
        self.results['forces'] = F_fin
        self.results['gap'] = gap_fin
        return Eat_fin, Etot_fin, F_fin, gap_fin
    
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    

class SEQM_singlepoint(torch.nn.Module):
    def __init__(self, seqm_settings=None):
        super(SEQM_singlepoint, self).__init__()
        self.settings = seqm_settings
        self.core_runner = SEQM_singlepoint_core(seqm_settings)
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
    
    def forward(self, p, species, coordinates, custom_params=[]):
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
        res = self.core_runner(p_sorted, Z, xyz, custom_params=custom_params)
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
        
    

class SEQM_multirun_core(torch.nn.Module):
    def __init__(self, species, coordinates, custom_params=[], 
                 seqm_settings=None):
        super(SEQM_multirun_core, self).__init__()
        settings = default_settings
        settings.update(seqm_settings)
        self.const = Constants()
        self.Z, self.xyz = species, coordinates
        self.xyz.requires_grad_(True)
        self.custom_par = custom_params
        elements = sorted(set([0] + self.Z.reshape(-1).tolist()))
        settings['elements'] = torch.tensor(elements)
        settings['learned'] = custom_params
        settings['eig'] = True
        self.calc = Energy(settings)
        my_parser = Parser(self.calc.seqm_parameters)
        n_occ = my_parser(self.const, self.Z, self.xyz)[4]
        self.homo = (n_occ-1).unsqueeze(-1)
        self.lumo = n_occ.unsqueeze(-1)
        self.results = {}
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__

#    @staticmethod
    def forward(self, p):
#    TODO: NEED CUSTOM BACKWARD FOR WHEN CALCULTION FAILS (RETURN NaN).
#          IS p.register_hook(lambda grad: grad * NaN) working?
#    def forward(self, ctx, p):
        """ Run calculation. """
        learnedpar = {par:p[i] for i, par in enumerate(self.custom_par)}
        try:
            res = self.calc(self.const, self.xyz, self.Z, learnedpar,
                            all_terms=True)
        except RuntimeError:
            p.register_hook(lambda grad: grad * torch.nan)
            self.xyz.register_hook(lambda grad: grad * torch.nan)
            return torch.nan, torch.nan, torch.nan*torch.ones_like(self.xyz), torch.nan
        masking = (~res[-1]).float()
        Eat_fin = res[0] * masking
        Etot_fin = res[1] * masking
        F = -agrad(res[1].sum(), self.xyz, create_graph=True)[0]
        F_fin = F * masking[...,None,None]
        ehomo = torch.gather(res[6], 1, self.homo).reshape(-1)
        elumo = torch.gather(res[6], 1, self.lumo).reshape(-1)
        gap_fin = (elumo - ehomo) * masking
        self.results['atomization'] = Eat_fin
        self.results['energy'] = Etot_fin
        self.results['forces'] = F_fin
        self.results['gap'] = gap_fin
        return Eat_fin, Etot_fin, F_fin, gap_fin
        
    def get_property(self, property_name):
        if not property_name in self.results:
            raise ValueError("Property '"+property_name+"' not available.")
        return self.results[property_name]
        
    
class SEQM_multirun(torch.nn.Module):
    def __init__(self, species, coordinates, custom_params=[],
                 seqm_settings=None):
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
                custom_params=custom_params, seqm_settings=seqm_settings)
    
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
        
    

