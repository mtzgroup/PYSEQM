#############################################################################
# Dataset and DataLoader routines for SEQC input and reference data         #
#  - AbstractLoader: Abstract base class as template for dataset objects    #
#                                                                           #
# Current (Feb/12)                                                          #
# TODO: . typing                                                            #
#############################################################################

import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .pyseqm_helpers import prepare_array


inp2int = {"species":"species",
           "coordinates":"coordinates",
           "atomization_ref":"Eat_ref",
           "atomization_weight":"Eat_weight",
           "energy_ref":"Etot_ref",
           "energy_weight":"Etot_weight",
           "forces_ref":"F_ref",
           "forces_weight":"F_weight",
           "gap_ref":"gap_ref",
           "gap_weight":"gap_weight",
          }
base_inps = [k for k in inp2int.keys()]
pad_inps = ["coordinates","forces_ref"]

class AbstractLoader(Dataset, ABC):
    def __init__(self, species, coordinates, atomization_ref=None, 
                 atomization_weight=1., energy_ref=None, energy_weight=1.,
                 forces_ref=None, forces_weight=1., gap_ref=None,
                 gap_weight=1.):
        super(AbstractLoader, self).__init__()
        for inp in base_inps:
            inp_obj = eval(inp)
            int_nam = inp2int[inp]
            if not torch.is_tensor(inp_obj):
                if inp in pad_inps or (isinstance(inp_obj, list) and torch.is_tensor(inp_obj[0])):
                    exec("self."+int_nam+" = pad_sequence("+inp+", batch_first=True)")
                    exec("self."+int_nam+".requires_grad_(False)")
                else:
                    exec("self."+int_nam+" = torch.tensor("+inp+", requires_grad=False)")
            else:
                exec("self."+int_nam+" = "+inp)
                exec("self."+int_nam+".requires_grad_(False)")
        self.nMols = self.species.shape[0]

    def __len__(self): return self.nMols

    @abstractmethod
    def __getitem__(self, idx): pass

    def dataloader(self, **kwargs):
        per_batch = kwargs.get("batch_size", 1)
        no_last = kwargs.get("drop_last", False)
        if (per_batch > self.nMols) and no_last:
            msg  = "Batch size larger than dataset. With `drop_last=True` "
            msg += "dataloader will return no items!"
            raise ValueError(msg)
        return DataLoader(self, **kwargs)




class SEQM_data(AbstractLoader):
    def __init__(self, species, coordinates, atomization_ref=None, 
                 atomization_weight=1., energy_ref=None, energy_weight=1.,
                 forces_ref=None, forces_weight=1., gap_ref=None,
                 gap_weight=1.):
        super(SEQM_data, self).__init__(species, coordinates,
                        atomization_ref=atomization_ref,
                        atomization_weight=atomization_weight,
                        energy_ref=energy_ref, energy_weight=energy_weight,
                        forces_ref=forces_ref, forces_weight=forces_weight,
                        gap_ref=gap_ref, gap_weight=gap_weight)
    
    def __getitem__(self, idx):
        inputs = (self.species[idx], self.coordinates[idx])
        refs = (self.Eat_ref[idx], self.Etot_ref[idx], self.F_ref[idx], self.gap_ref[idx])
        weights = (self.Eat_weight, self.Etot_weight, self.F_weight, self.gap_weight)
        return inputs, refs, weights
    

class AMASE_data(AbstractLoader):
    def __init__(self, species, coordinates, desc, atomization_ref=None, 
                 atomization_weight=1., energy_ref=None, energy_weight=1.,
                 forces_ref=None, forces_weight=1., gap_ref=None,
                 gap_weight=1.):
        super(AMASE_data, self).__init__(species, coordinates,
                        atomization_ref=atomization_ref,
                        atomization_weight=atomization_weight,
                        energy_ref=energy_ref, energy_weight=energy_weight,
                        forces_ref=forces_ref, forces_weight=forces_weight,
                        gap_ref=gap_ref, gap_weight=gap_weight)
        if not torch.is_tensor(desc) or (isinstance(desc, list) and torch.is_tensor(desc[0])):
            self.desc = pad_sequence(desc, batch_first=True)
        self.desc.requires_grad_(False)
        
    def __getitem__(self, idx):
        inputs = (self.species[idx], self.coordinates[idx], self.desc[idx])
        refs = (self.Eat_ref[idx], self.Etot_ref[idx], self.F_ref[idx], self.gap_ref[idx])
        weights = (self.Eat_weight, self.Etot_weight, self.F_weight, self.gap_weight)
        return inputs, refs, weights
    

