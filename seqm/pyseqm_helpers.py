import torch
import numpy as np
from inspect import getfullargspec
from ase.data import atomic_numbers
from seqm.basics import Parser, Energy
from seqm.seqm_functions.parameters import params


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
device = torch.device('cuda') if has_cuda else torch.device('cpu')


default_settings = {
   'method'             : 'AM1', 
   'scf_eps'            : 1.0e-6, 
   'scf_converger'      : [2,0.0], 
   'sp2'                : [True, 1e-5],
   'parameter_file_dir' : '/home/martin/work/software/PYSEQM/seqm/params/',
   'pair_outer_cutoff'  : 1.0e10, 
   'Hf_flag'            : False,
   }


class pyseqm_orderator:
    """
    Class to handle inputs and output of pyseqm calculation.
    
    Parameters
    ----------
    species : ndarray (int or str), shape (nAtoms,)
        atomic numbers or chemical symbols
    coordinates : ndarray, shape (nAtoms, 3)
        atomic positions
    """
    
    def __init__(self, species, coordinates):
        n = len(species)
        if isinstance(species, np.ndarray): species = species.tolist()
        self.species = species.copy()
        for i, s in enumerate(self.species):
            if type(s) is str: self.species[i] = atomic_numbers[s]
        ## sort by descending atomic number (and store reverse order)
        self.sortidx = np.argsort(self.species, kind='stable')[::-1]
        self.sortidx = self.sortidx.tolist()
        self.reverse = [self.sortidx.index(i) for i in range(n)]
        self.coordinates = coordinates.copy()
        if isinstance(self.coordinates, np.ndarray):
            self.coordinates = self.coordinates.tolist()
    
    def prepare_input(self, *args):
        """
        Prepare structural input for pyseqm calculation.
        
        Returns
        -------
        species : ndarray, shape (nAtoms,)
            atomic numbers in descending order
        coordinates : ndarray, shape (nAtoms,3)
            atomic positions according to species output
        any array provided in args ordered according to new atom ordering
        """
        Z = [self.species[i] for i in self.sortidx]
        Z = torch.as_tensor([Z], dtype=torch.int64, device=device)
        xyz = [self.coordinates[i] for i in self.sortidx]
        xyz = torch.as_tensor([xyz], device=device)
        rest = []
        for arg in args:
            if type(arg) in [np.ndarray, torch.Tensor]:
                arg = arg.tolist()
            s_arg = [arg[j] for j in self.sortidx]
            rest.append( torch.as_tensor(s_arg, device=device) )
        return Z, xyz, *args
    
    def reorder_output(self, output, *args):
        """
        Restores ordering of any provided array according to original atom ordering
        
        Parameters
        ----------
        output : ndarray, shape (nAtoms, ...)
            array of atomic property obtained from pyseqm calculation
        any number of arrays to be reordered
                
        Returns
        -------
        output : ndarray, shape (nAtoms, ...)
            reordered input array according to original atom ordering
        any array provided in args re-ordered according to original atom ordering

        """
        output = output[self.reverse]
        if len(args)>0:
            new_args = []
            for arg in args: new_args.append( arg[self.reverse] )
            return output, *new_args
        return output
        

def get_energy_calculator(species, coordinates, custom_parameters=[], **kwargs):
    """
    Creates a pyseqm energy calculator instance
    """
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    settings = default_settings.copy()
    settings['elements'] = elements
    settings['learned'] = custom_parameters
    for key, val in kwargs.items(): settings[key] = val
    calc = Energy(settings).to(device)
    return calc
    

def get_default_parameters(species, method, parameter_dir, param_list):
    """
    Returns default values of specified parameters for atoms in system
    """
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    default_p = params(method=method, elements=elements, \
                       root_dir=parameter_dir, \
                       parameters=param_list).to(device)
    default_p = default_p[species][0].transpose(0,1).contiguous()
    return default_p
    
def get_ordered_args(func, **kwargs):
    """
    Returns kwarg input to function `func` ordered for position arg input 
    with default values where kwarg is missing.
    
    Parameters:
    -----------
    func : callable
        Function to check for ordered arg input and defaults
    **kwargs : `key=arg` structure
        Possible kwarg input to `func`
    
    Returns:
    --------
    ordered list of input for positional args of `func`
    """
    argspec = getfullargspec(func)
    func_defaults = argspec.defaults
    func_kwargs = argspec.args[-len(func_defaults):]
    ordered_args = []
    for i, key in enumerate(func_kwargs):
        ordered_args.append(kwargs.get(key,func_defaults[i]))
    return tuple(ordered_args)

    
