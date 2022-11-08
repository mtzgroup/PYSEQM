import torch
import numpy as np
from inspect import getfullargspec
from scipy.optimize import approx_fprime
from ase.data import atomic_numbers
from seqm.basics import Energy
from seqm.seqm_functions.parameters import params


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device('cuda')
    sp2_def = [True, 1e-5]
else:
    device = torch.device('cpu')
    sp2_def = [False]



default_settings = {
   'method'             : 'AM1', 
   'scf_eps'            : 1.0e-6, 
   'scf_converger'      : [2,0.0], 
   'sp2'                : sp2_def,
   'pair_outer_cutoff'  : 1.0e10, 
   'Hf_flag'            : False,
   }

default_bounds = {
  'AM1': {
    'U_ss'       : [-640. ,   0.], 
    'U_pp'       : [-640. ,   0.], 
    'zeta_s'     : [   0.5,  64.], 
    'zeta_p'     : [   0.5,  42.], 
    'zeta_d'     : [   0.5,  42.],
    'beta_s'     : [-420. ,   0.], 
    'beta_p'     : [-420. ,   0.], 
    'g_ss'       : [   0. , 420.], 
    'g_sp'       : [   0. , 240.], 
    'g_pp'       : [   0. , 240.], 
    'g_p2'       : [   0. , 240.], 
    'h_sp'       : [   0. ,  64.], 
    'alpha'      : [   0. ,  64.], 
    'Gaussian1_K': [ -16. ,  16.], 
    'Gaussian1_L': [   0. ,  96.], 
    'Gaussian1_M': [   0. ,  64.], 
    'Gaussian2_K': [ -16. ,  16.], 
    'Gaussian2_L': [   0. , 128.], 
    'Gaussian2_M': [   0. ,  64.], 
    'Gaussian3_K': [  -8. ,  16.], 
    'Gaussian3_L': [   0. , 128.], 
    'Gaussian3_M': [   0. ,  64.], 
    'Gaussian4_K': [  -8. ,   8.], 
    'Gaussian4_L': [ -16. ,  96.], 
    'Gaussian4_M': [ -16. ,  72.],
  },
  'PM3': {
    'U_ss'       : [-640. ,   0.],
    'U_pp'       : [-640. ,   0.],
    'U_dd'       : [-640. ,   0.],
    'zeta_s'     : [   0.5,  64.],
    'zeta_p'     : [   0.5,  42.],
    'zeta_d'     : [   0.5,  42.],
    'beta_s'     : [-420. ,   0.],
    'beta_p'     : [-420. ,   0.],
    'g_ss'       : [   0. , 420.],
    'g_sp'       : [   0. , 240.],
    'g_pp'       : [   0. , 240.],
    'g_p2'       : [   0. , 240.],
    'h_sp'       : [   0. ,  64.],
    'alpha'      : [   0. ,  64.],
    'Gaussian1_K': [ -16. ,  16.],
    'Gaussian1_L': [   0. ,  96.],
    'Gaussian1_M': [   0. ,  64.],
    'Gaussian2_K': [ -16. ,  16.],
    'Gaussian2_L': [   0. , 128.],
    'Gaussian2_M': [   0. ,  64.],
    'Gaussian3_K': [  -8. ,  16.],
    'Gaussian3_L': [   0. , 128.],
    'Gaussian3_M': [   0. ,  64.],
    'Gaussian4_K': [  -8. ,   8.],
    'Gaussian4_L': [ -16. ,  96.],
    'Gaussian4_M': [ -16. ,  72.],
  },
  'MNDO': {
    'U_ss'       : [-640. ,   0.],
    'U_pp'       : [-640. ,   0.],
    'U_dd'       : [-640. ,   0.],
    'zeta_s'     : [   0.5,  42.],
    'zeta_p'     : [   0.5,  42.],
    'zeta_d'     : [   0.5,  42.],
    'beta_s'     : [-420. ,   0.],
    'beta_p'     : [-420. ,   0.],
    'g_ss'       : [   0. , 240.],
    'g_sp'       : [   0. , 240.],
    'g_pp'       : [   0. , 240.],
    'g_p2'       : [   0. , 240.],
    'h_sp'       : [   0. ,  64.],
    'alpha'      : [   0. ,  64.],
    'polvom'     : [   0. ,  64.],
  },
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
    Creates a pyseqm calculator instance from `module`.
    """
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    seqm_settings = default_settings.copy()
    seqm_settings['elements'] = elements
    seqm_settings['learned'] = custom_parameters
    seqm_settings.update(kwargs)
    calc = Energy(seqm_settings).to(device)
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

   
def post_process_result(p, p_init, loss_func, loss_arguments, nAtoms):
    p_ref = np.reshape(p_init,(-1,nAtoms))
    p_opt = np.reshape(p,(-1,nAtoms))
    dp = p_opt - p_ref
    gradL = approx_fprime(p, loss_func, 1.49e-7, *loss_arguments)
    gradL = np.reshape(gradL,(-1,nAtoms))
    loss_init = loss_func(p_init, *loss_arguments)
    loss_opt = loss_func(p, *loss_arguments)
    dloss = loss_opt - loss_init
    return p_opt, dp, gradL, loss_opt, dloss
    

def write_param_summary(p, dp, loss_opt, dloss, pname_list, symbols, 
                        writer='stdout', ID='#OPTIMIZED', close_after=False):
    nAtoms = len(symbols)
    if writer == 'stdout':
        from sys import stdout
        writer = stdout.writelines
    elif type(writer) is str:
        f = open(writer, 'a')
        writer = f.write
        close_after = True
    elif not callable(writer):
        msg  = "Input 'writer' has to be 'stdout', file_name, or "
        msg += "print/write function."
        raise RuntimeError(msg)
    
    writer(ID+'\n')
    if (0.01 <= abs(dloss) < 10.):
        dlstr = '({0: 5.2f})'.format(dloss)
    else: 
        dlstr = '({0: 5.2e})'.format(dloss)
    lstr  = '#LOSS (change from default):   '
    lstr += '{0: 6.3e} '.format(loss_opt)+dlstr
    writer(lstr+'\n')
    writer('---------------------------------------\n')
    writer('#PARAM_OPT: Current optimal parameters\n')
    sstr  = '                 '
    sstr += ''.join(['  {0:6s}   '.format(s) for s in symbols])
    writer(sstr[:-2]+'\n')
    for i, pname in enumerate(pname_list):
        pstr  = ' {0:<12s}: '.format(pname)
        pstr += ''.join(['{0: 9.4f}  '.format(p[i,j]) for j in range(nAtoms)])
        writer(pstr[:-2]+'\n')
    writer('---------------------------------------\n')
    writer('#DELTA_PARAM: Change from default parameters\n')
    sstr  = '                 '
    sstr += ''.join(['  {0:6s}   '.format(s) for s in symbols])
    writer(sstr[:-2]+'\n')
    for i, pname in enumerate(pname_list):
        pstr  = ' {0:<12s}: '.format(pname)
        pstr += ''.join(['{0: 9.4f}  '.format(dp[i,j]) for j in range(nAtoms)])
        writer(pstr[:-2]+'\n')
    if close_after:
        if hasattr(writer, 'close'): writer.close()
    return

    
def write_gradient_summary(gradL, symbols, pname_list, writer='stdout', close_after=False):
    if writer == 'stdout':
        from sys import stdout
        writer = stdout.writelines
    elif type(writer) is str:
        f = open(writer, 'a')
        writer = f.write
        close_after = True
    elif not callable(writer):
        raise RuntimeError("Input 'writer' has to be 'stdout', file_name, or print/write function.")
    
    writer('---------------------------------------\n')
    writer('#DLOSS_DPARAM: Gradient of current loss\n')
    sstr  = '                  '
    sstr += ''.join(['  {0:6s}   '.format(s) for s in symbols])
    writer(sstr[:-2]+'\n')
    for i, pname in enumerate(pname_list):
        gstr  = ' {0:<12s}: '.format(pname)
        gstr += ''.join(['{0: 5.2e}  '.format(g) for g in gradL[i]])
        writer(gstr[:-2]+'\n')
    if close_after:
        if hasattr(writer, 'close'): writer.close()
    return
    

#--EOF--# 
