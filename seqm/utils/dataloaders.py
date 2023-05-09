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
from torch.utils.data._utils.collate import collate
from h5py4mol import dataloader as h5pyloader


weight_inps = ["atomization_weight", "energy_weight", "forces_weight",
               "gap_weight"]

class AbstractLoader(Dataset, ABC):
    def __init__(self, hdf5_file, labels, atomization_weight=1., 
                 energy_weight=1., forces_weight=1., gap_weight=1.,
                 read_desc=False):
        super(AbstractLoader, self).__init__()
        self.nMols = len(labels)
        for w_inp in weight_inps: exec("self."+w_inp+" = "+w_inp)
        Z, xyz, desc, Eat, Etot, F, gap = [], [], [], [], [], [], []
        for l in labels:
            with h5pyloader(hdf5_file) as dl: my_data = dl.get_data(l)
            Z.append(torch.tensor(my_data['atomic_numbers'], requires_grad=False))
            xyz.append(torch.tensor(my_data['positions']))
            Eat.append(my_data['dft_atomization'])
            Etot.append(my_data['dft_energy'])
            F.append(torch.tensor(my_data['dft_forces'], requires_grad=False))
            gap.append(my_data['dft_homo_lumo_gap'])
            if read_desc: desc.append(my_data['SOAP0'])
        self.species = pad_sequence(Z, batch_first=True)
        self.species.requires_grad_(False)
        self.coordinates = pad_sequence(xyz, batch_first=True)
        if read_desc:
            self.desc = pad_sequence(desc, batch_first=True)
            self.desc.requires_grad_(False)
        self.Eat_ref = torch.tensor(Eat, requires_grad=False)
        self.Etot_ref = torch.tensor(Etot, requires_grad=False)
        self.F_ref = pad_sequence(F, batch_first=True)
        self.F_ref.requires_grad_(False)
        self.gap_ref = torch.tensor(gap, requires_grad=False)
    
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
    def __init__(self, hdf5_file, labels, atomization_weight=1.,
                 energy_weight=1., forces_weight=1., gap_weight=1.):
        super(SEQM_data, self).__init__(hdf5_file, labels,
            atomization_weight=atomization_weight, energy_weight=energy_weight,
            forces_weight=forces_weight, gap_weight=gap_weight)
    
    def __getitem__(self, idx):
        inputs = (self.species[idx], self.coordinates[idx])
        refs = (self.Eat_ref[idx], self.Etot_ref[idx], self.F_ref[idx], self.gap_ref[idx])
        weights = (self.atomization_weight, self.energy_weight, self.forces_weight, self.gap_weight)
        return inputs, refs, weights
    

class AMASE_data(AbstractLoader):
    def __init__(self, hdf5_file, labels, atomization_weight=1., 
                 energy_weight=1., forces_weight=1., gap_weight=1.):
        super(AMASE_data, self).__init__(hdf5_file, labels,
            atomization_weight=atomization_weight, energy_weight=energy_weight,
            forces_weight=forces_weight, gap_weight=gap_weight, read_desc=True)
        
    def __getitem__(self, idx):
        inputs = (self.species[idx], self.coordinates[idx], self.desc[idx])
        refs = (self.Eat_ref[idx], self.Etot_ref[idx], self.F_ref[idx], self.gap_ref[idx])
        weights = (self.atomization_weight, self.energy_weight, self.forces_weight, self.gap_weight)
        return inputs, refs, weights
    


def pad_tensor_fn(batch, *, collate_fn_map):
    """ Padding of tensor input for batched mode runs. """
    return pad_sequence(batch, batch_first=True)

def collate_floats_fn(batch, *, collate_fn_map):
    """ Collates floats from batch into tensor. """
    return torch.tensor(batch)

def custom_collate_fn(batch):
    """
    Custom collate function to get list of database entries into required
    shape and format (zero-padded tensors for PYSEQM batch mode).
    """
    collate_map = {torch.Tensor:pad_tensor_fn, float:collate_floats_fn}
    return collate(batch, collate_fn_map=collate_map)
    

class AbstractHDF5Loader(Dataset, ABC):
    """
    Abstract base class to load data from hdf5 file.
    `__get_item__` has to be implemented by child classes.
    
    Parameters:
    -----------
    hdf5_file: str
        file name / location of hdf5 file
    labels: list of str
        names of database entries
    *_weight: float
        weight of property when included in loss function
        (this will be deprecated in future version!)
    """
    def __init__(self, hdf5_file, labels=[], atomization_weight=1.,
                 energy_weight=1., forces_weight=1., gap_weight=1.):
        super(AbstractHDF5Loader, self).__init__()
        self.db_file = hdf5_file
        self.nMols = len(labels)
        self.db_labels = labels
        for w_inp in weight_inps: exec("self."+w_inp+" = "+w_inp)
        
    def __len__(self): return self.nMols
    
    @abstractmethod
    def __getitem__(self, idx): pass
    
    def dataloader(self, **kwargs):
        """ Provide pytorch dataloader object for database. """
        per_batch = kwargs.get("batch_size", 1)
        no_last = kwargs.get("drop_last", False)
        if (per_batch > self.nMols) and no_last:
            msg  = "Batch size larger than dataset. With `drop_last=True` "
            msg += "dataloader will return no items!"
            raise ValueError(msg)
        return DataLoader(self, collate_fn=custom_collate_fn, num_workers=1,
                          prefetch_factor=2, **kwargs)
        
    

class SEQM_HDF5data(AbstractHDF5Loader):
    def __init__(self, hdf5_file, labels=[], atomization_weight=1.,
                 energy_weight=1., forces_weight=1., gap_weight=1.):
        super(SEQM_HDF5data, self).__init__(hdf5_file, labels=labels,
                                    atomization_weight=1., energy_weight=1.,
                                    forces_weight=1., gap_weight=1.)
    
    def __getitem__(self, idx):
        with h5pyloader(self.db_file) as dl:
            my_data = dl.get_data(self.db_labels[idx])
        Z = torch.tensor(my_data['atomic_numbers'], requires_grad=False)
        xyz = torch.tensor(my_data['positions'])
        Eat = my_data['dft_atomization']
        Etot = my_data['dft_energy']
        F = torch.tensor(my_data['dft_forces'], requires_grad=False)
        gap = my_data['dft_homo_lumo_gap']
        inputs = (Z, xyz)
        refs = (Eat, Etot, F, gap)
        weights = (self.atomization_weight, self.energy_weight, self.forces_weight, self.gap_weight)
        return inputs, refs, weights
        
    
class AMASE_HDF5data(AbstractHDF5Loader):
    def __init__(self, hdf5_file, labels=[], atomization_weight=1.,
                 energy_weight=1., forces_weight=1., gap_weight=1.):
        super(AMASE_HDF5data, self).__init__(hdf5_file, labels=labels,
                                    atomization_weight=1., energy_weight=1.,
                                    forces_weight=1., gap_weight=1.)

    def __getitem__(self, idx):
        with h5pyloader(self.db_file) as dl:
            my_data = dl.get_data(self.db_labels[idx])
        Z = torch.tensor(my_data['atomic_numbers'], requires_grad=False)
        xyz = torch.tensor(my_data['positions'])
        Eat = my_data['dft_atomization']
        Etot = my_data['dft_energy']
        F = torch.tensor(my_data['dft_forces'], requires_grad=False)
        gap = my_data['dft_homo_lumo_gap']
        desc = torch.tensor(my_data['SOAP0'], requires_grad=False)
        inputs = (Z, xyz, desc)
        refs = (Eat, Etot, F, gap)
        weights = (self.atomization_weight, self.energy_weight, self.forces_weight, self.gap_weight)
        return inputs, refs, weights
    

