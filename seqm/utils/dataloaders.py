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


class AbstractLoader(ABC, Dataset):
    def __init__(self, species, coordinates, atomization_ref=None, 
                 atomization_weight=1., energy_ref=None, energy_weight=1.,
                 forces_ref=None, forces_weight=1., gap_ref=None,
                 gap_weight=1.):
        super(AbstractLoader, self).__init__()
        self.species = pad_sequence(species, batch_first=True)
        self.nMols = self.species.shape[0]
        self.coordinates = pad_sequence(coordinates, batch_first=True)
        self.Eat_ref = torch.tensor(atomization_ref)
        self.Etot_ref = torch.tensor(energy_ref)
        self.F_ref = pad_sequence(forces_ref, batch_first=True)
        self.gap_ref = torch.tensor(gap_ref)
        self.Eat_weight = torch.tensor(atomization_weight)
        self.Etot_weight = torch.tensor(energy_weight)
        self.F_weight = torch.tensor(forces_weight)
        self.gap_weight = torch.tensor(gap_weight)

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
        self.desc = pad_sequence(desc, batch_first=True)
        
    def __getitem__(self, idx):
        inputs = (self.species[idx], self.coordinates[idx], self.desc[idx])
        refs = (self.Eat_ref[idx], self.Etot_ref[idx], self.F_ref[idx], self.gap_ref[idx])
        weights = (self.Eat_weight, self.Etot_weight, self.F_weight, self.gap_weight)
        return inputs, refs, weights
    

