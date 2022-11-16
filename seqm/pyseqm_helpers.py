import torch
import numpy as np
from inspect import getfullargspec
from scipy.optimize import approx_fprime, LinearConstraint, Bounds
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

default_multi_bound = {
    'U_ss'       : [  0.0,  5.0], 
    'U_pp'       : [  0.0,  5.0], 
    'U_dd'       : [  0.0,  5.0],
    'zeta_s'     : [  0.5,  5.0], 
    'zeta_p'     : [  0.5,  5.0], 
    'zeta_d'     : [  0.5,  5.0],
    'beta_s'     : [  0.0,  5.0], 
    'beta_p'     : [  0.0,  5.0], 
    'beta_d'     : [  0.0,  5.0],
    'g_ss'       : [  0.0,  5.0], 
    'g_sp'       : [  0.0,  5.0], 
    'g_pp'       : [  0.0,  5.0], 
    'g_p2'       : [  0.0,  5.0], 
    'h_sp'       : [  0.0, 20.0], 
    'alpha'      : [  0.0, 10.0], 
    'Gaussian1_K': [-10.0, 10.0], 
    'Gaussian1_L': [  0.0, 10.0], 
    'Gaussian1_M': [  0.0, 20.0], 
    'Gaussian2_K': [-10.0, 10.0], 
    'Gaussian2_L': [  0.0, 20.0], 
    'Gaussian2_M': [  0.0, 20.0], 
    'Gaussian3_K': [-20.0, 20.0], 
    'Gaussian3_L': [  0.0, 10.0], 
    'Gaussian3_M': [  0.0, 20.0], 
    'Gaussian4_K': [-40.0, 40.0], 
    'Gaussian4_L': [  0.0, 20.0], 
    'Gaussian4_M': [  0.0, 20.0],
}

parameter_names = {
  'AM1': ['U_ss', 'U_pp', 'zeta_s', 'zeta_p', 'zeta_d', 'beta_s',
          'beta_p', 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha',
          'Gaussian1_K', 'Gaussian1_L', 'Gaussian1_M',
          'Gaussian2_K', 'Gaussian2_L', 'Gaussian2_M',
          'Gaussian3_K', 'Gaussian3_L', 'Gaussian3_M',
          'Gaussian4_K', 'Gaussian4_L', 'Gaussian4_M'],
  'PM3': ['U_ss', 'U_pp', 'U_dd', 'zeta_s', 'zeta_p', 'zeta_d', 'beta_s',
          'beta_p', 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha',
          'Gaussian1_K', 'Gaussian1_L', 'Gaussian1_M',
          'Gaussian2_K', 'Gaussian2_L', 'Gaussian2_M'],
  'MNDO': ['U_ss', 'U_pp', 'U_dd', 'zeta_s', 'zeta_p', 'zeta_d', 'beta_s',
           'beta_p', 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha',
           'polvom']
}

max_defaults = {
  'AM1': {'U_ss':-136.11, 'U_pp':-104.89, 'zeta_s':3.77, 'zeta_p':2.52, 
          'zeta_d':1.0, 'beta_s':-69.59, 'beta_p':-29.27, 'g_ss':59.42, 
          'g_sp':17.25, 'g_pp':16.71, 'g_p2':14.91, 'h_sp':4.83, 
          'alpha':6.02, 'Gaussian1_K':1.75, 'Gaussian1_L':12.39, 
          'Gaussian1_M':2.05, 'Gaussian2_K':0.9, 'Gaussian2_L':10.79, 
          'Gaussian2_M':3.2, 'Gaussian3_K':0.04, 'Gaussian3_L':13.56, 
          'Gaussian3_M':3.01, 'Gaussian4_K':-0.01, 'Gaussian4_L':5.0, 
          'Gaussian4_M':2.65},
  'PM3': {'U_ss':-116.62, 'U_pp':-105.69, 'U_dd':-100.0, 'zeta_s':7.0, 
          'zeta_p':2.5, 'zeta_d':2.88, 'beta_s':-48.41, 'beta_p':-27.75, 
          'g_ss':16.01, 'g_sp':16.07, 'g_pp':14.82, 'g_p2':16.0, 
          'h_sp':4.04, 'alpha':3.36, 'Gaussian1_K':3.0, 'Gaussian1_L':6.5, 
          'Gaussian1_M':2.32, 'Gaussian2_K':-2.55, 'Gaussian2_L':6.5, 
          'Gaussian2_M':2.97},
  'MNDO': {'U_ss':-131.07, 'U_pp':-105.78, 'U_dd':-100.0, 'zeta_s':4.0, 
           'zeta_p':2.85, 'zeta_d':1.0, 'beta_s':-48.29, 'beta_p':-36.51, 
           'g_ss':16.92, 'g_sp':17.25, 'g_pp':16.71, 'g_p2':14.91, 
           'h_sp':4.83, 'alpha':3.42, 'polvom':4.09}
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
        

def get_energy_calculator(species, coordinates, custom_parameters=(), **kwargs):
    """
    Creates a pyseqm calculator instance from `module`.
    """
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    seqm_settings = default_settings.copy()
    seqm_settings['elements'] = elements
    seqm_settings['learned'] = custom_parameters
    seqm_settings['eig'] = True
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
    
def get_par_names_bounds_defaults(species, method, parameter_dir, 
                                  change_zeros=False):
    """
    Returns all parameter names, corresponding default values, standard 
    bounds (spanning wide space), constraints (physically-reasonable space).
        Bounds are given as multiples of default values
    If change_zeros: Extend bounds for entries that have 0. as default value
        (These should be irrelevant for the system and might only complicate,
         for example, optimization, but becomes relevant when extending the
         default parametrization.)
    
    Parameters:
    -----------
    species : torch.Tensor / array-like, shape (nAtoms,)
        atomic numbers
    method : str
        type of NDDO calculation to perform [AM1, PM3, MNDO]
    parameter_dir : str
        path to directory with default parameters
    change_zeros : bool (default False)
        whether or not to change parameters with zero default value
    """
    pnames = parameter_names[method]
    nA = species.size()[-1]
    bounds = [default_multi_bound[par] for par in pnames]
    bounds_expanded = np.array([[b,]*nA for b in bounds]).T
    lowupp = np.array([bounds_expanded[0].T.flatten(),
                       bounds_expanded[1].T.flatten()])
    pdef = get_default_parameters(species, method, parameter_dir, pnames)
    def4b = pdef.clone().detach().numpy()
    def4c = def4b.copy().flatten()
    zero_idx = np.argwhere(np.abs(def4b)<1e-8)
    ## bounds in some optimizers cannot be equal (fixed parameter)
    ## as this causes problem in mapping parameter to [0;1)
    ## we use placeholder bounds and introduce constraints to fix values
    for ij in zero_idx:
        def4b[tuple(ij)] = max_defaults[method][pnames[ij[0]]]
    def4b = def4b.flatten()
    low_b, upp_b = np.sort(lowupp * def4b, axis=0)
    low_b = np.minimum(low_b, def4c)
    upp_b = np.maximum(upp_b, def4c)
    pbounds = Bounds(low_b, upp_b)
    if change_zeros: def4c = def4b
    low_c, upp_c = np.sort(lowupp * def4c, axis=0)
    pconstr = LinearConstraint(np.eye(low_c.size), low_c, upp_c)
    return tuple(pnames), pdef, pbounds, pconstr
    

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

   
def post_process_result(p, p_init, loss_func, nAtoms):
    p_ref = np.reshape(p_init,(-1,nAtoms))
    p_opt = np.reshape(p,(-1,nAtoms))
    dp = p_opt - p_ref
    gradL = approx_fprime(p, loss_func, 1.49e-7)
    gradL = np.reshape(gradL,(-1,nAtoms))
    loss_init = loss_func(p_init)
    if type(loss_init) in [np.ndarray, list]:
        if np.size(loss_init) == 1: loss_init = float(loss_init)
    loss_opt = loss_func(p)
    if type(loss_opt) in [np.ndarray, list]:
        if np.size(loss_opt) == 1: loss_opt = float(loss_opt)
    dloss = loss_opt - loss_init
    return p_opt, dp, gradL, loss_opt, dloss
    

def get_writer(writer_in, closing_in=False):
    if writer_in == 'stdout':
        from sys import stdout
        writer_in = stdout.writelines
    elif type(writer_in) is str:
        f = open(writer_in, 'a')
        writer_in = f.write
        closing_in = True
    elif not callable(writer_in):
        msg  = "Input 'writer' has to be 'stdout', file_name, or "
        msg += "print/write function."
        raise RuntimeError(msg)
    return writer_in, closing_in
    

def write_param_summary(p, dp, loss_opt, dloss, pname_list, symbols, 
                        writer='stdout', ID='#OPTIMIZED', close_after=False):
    nAtoms = len(symbols)
    writer, close_after = get_writer(writer, close_after)
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
    writer, close_after = get_writer(writer, close_after)
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
