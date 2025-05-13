import torch
from collections.abc import Sequence
from warnings import warn
from seqm.basics import *
from .basics import Parser


mol_attributes = ["force", "velocities", "acc", "dm", "q", "d", "Hf", "Etot", "Eelec",
                  "Enuc", "Eiso", "e_mo", "e_gap", "mo_coeff", "charge", "dipole",
                  "Electronic_entropy", "Fermi_occ", "dP2dt2", "Krylov_Error"]

class Molecule(torch.nn.Module,Sequence):
    def __init__(self, const, seqm_parameters, coordinates, species, charges=0, mult=1, *args, **kwargs):
        """
        unit for timestep is femtosecond
        output: [molecule id list, frequency N, prefix]
            molecule id in the list are output, staring from 0 to nmol-1
            geometry is writted every dump step to the file with name prefix + molid + .xyz
            step, temp, and total energy is print to screens for select molecules every thermo
        """
        super(Molecule, self).__init__(*args, **kwargs)
        self.const = const
        self.seqm_parameters = seqm_parameters
        self.coordinates = coordinates
        self.species = species
        if not torch.is_tensor(charges):
            charges = charges * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.tot_charge = charges
        if not torch.is_tensor(mult):
            mult = mult * torch.ones(coordinates.size()[0], device=coordinates.device)
        self.mult = mult
        
        MASS = torch.as_tensor(self.const.mass)
        # put the padding virtual atom mass finite as for accelaration, F/m evaluation.
        MASS[0] = 1.0
        self.mass = MASS[self.species].unsqueeze(2)
        for attr in mol_attributes: setattr(self, attr, None)
        self.init_properties(self.seqm_parameters, *args, **kwargs)
        
                
    def init_properties(self, seqm_parameters, *args, **kwargs):
        self.parser = Parser(seqm_parameters)
        self.nmol, self.molsize, \
        self.nHeavy, self.nHydro, self.nocc, \
        self.Z, self.maskd, self.atom_molid, \
        self.mask, self.mask_l, self.pair_molid, \
        self.ni, self.nj, self.idxi, self.idxj, self.xij, self.rij = self.parser(self, return_mask_l=True, *args, **kwargs)
        real_elements = self.Z.unique()
        elms_sorted = real_elements.sort(descending=True)[0].tolist()
        self.element_map = torch.tensor([elms_sorted.index(z) for z in self.Z], requires_grad=False)
        
    
    def get_coordinates(self):
        return self.coordinates

    def get_species(self):
        return self.species
    
    def __getitem__(self, idx):
        if not torch.is_tensor(idx): idx = torch.as_tensor(idx, dtype=torch.int64)
        i = torch.atleast_1d(idx)
        submol = type(self)(self.const, self.seqm_parameters, self.coordinates[i],
                            self.species[i], charges=self.tot_charge[i],
                            mult=self.mult[i])
        for attr in mol_attributes:
            if getattr(self, attr) is not None:
                setattr(submol, attr, getattr(self, attr)[i])
        return submol
    
    def __setitem__(self, idx, mol):
        if not torch.is_tensor(idx): idx = torch.as_tensor(idx, dtype=torch.int64)
        i = torch.atleast_1d(idx)
        s, c = self.species.clone(), self.coordinates.clone()
        s[i], c[i] = mol.species, mol.coordinates
        self.species, self.coordinates = s, c
        if self.species.requires_grad: self.species.retain_grad()
        if self.coordinates.requires_grad: self.coordinates.retain_grad()
        
        self.nmol, self.molsize, \
        self.nHeavy, self.nHydro, self.nocc, \
        self.Z, self.maskd, self.atom_molid, \
        self.mask, self.mask_l, self.pair_molid, \
        self.ni, self.nj, self.idxi, self.idxj, self.xij, self.rij = self.parser(self, return_mask_l=True)
        
        real_elements = self.Z.unique()
        elms_sorted = real_elements.sort(descending=True)[0].tolist()
        self.element_map = torch.tensor([elms_sorted.index(z) for z in self.Z], requires_grad=False)
        for attr in mol_attributes:
            check1 = (getattr(self, attr) is None) and (getattr(mol, attr) is not None)
            check2 = (getattr(self, attr) is not None) and (getattr(mol, attr) is None)
            if check1 or check2:
                warn("Attribute '"+attr+"' changed from/to None between previous and updated molecule. Updating to None.")
                setattr(self, attr, None)
            elif (getattr(self, attr) is not None) and (getattr(mol, attr) is not None):
                exec("self." + attr + "[i] = getattr(mol, attr)")
    
    def __len__(self):
        return self.nmol
    
#    def __eq__(self, other):
#        if not isinstance(other, type(self)): return False
#        dict1, dict2 = self.__dict__, other.__dict__
#        if dict1.keys() != dict2.keys(): return False
#        for key, attr in dict1.items():
#            if torch.is_tensor(attr):
#                try:
#                    if not torch.allclose(attr, dict2[key]): return False
#                except RuntimeError:
#                    return False
#            elif attr != dict2[key]:
#                print(key, attr, dict2[key])
#                return False
#        return True
    
    
