import torch
from itertools import chain
from .pyseqm_helpers import prepare_array, Orderator
from .seqm_core_runners import SEQM_singlepoint_core, SEQM_multirun_core
from .kernel_core_runners import AMASE_singlepoint_core, AMASE_multirun_core



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
        
    

