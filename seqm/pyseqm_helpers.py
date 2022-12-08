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


def scale_from_unit_cube(x, bounds, for_gradient=False):
    """
    Rescale input vector to the unit hypercube within the specified bounds.
    """
    shift = 0.5 * (bounds.ub + bounds.lb)
    span = np.fabs(bounds.ub - bounds.lb)
    if for_gradient: return x * span
    return shift + (x - 0.5) * span

def scale_to_unit_cube(x, bounds, for_gradient=False):
    """
    Rescale input vector to the unit hypercube within the specified bounds.
    """
    shift = 0.5 * (bounds.ub + bounds.lb)
    signed_span = bounds.ub - bounds.lb
    span_mask = np.abs(signed_span) < 1e-12
    span = np.where(span_mask, 1., np.fabs(signed_span))
    if for_gradient: return np.where(span_mask, 0., x/span)
    return np.where(span_mask, 1., (x - shift) / span + 0.5)
    

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
    
    def prepare_input(self, *args, axis=0):
        """
        Prepare structural input for pyseqm calculation.
        
        Returns
        -------
        species : ndarray, shape (nAtoms,)
            atomic numbers in descending order
        coordinates : ndarray, shape (nAtoms,3)
            atomic positions according to species output
        any array provided in args ordered according to new atom 
            ordering (sorted along `axis`)
        """
        Z = [self.species[i] for i in self.sortidx]
        Z = torch.as_tensor([Z], dtype=torch.int64, device=device)
        xyz = [self.coordinates[i] for i in self.sortidx]
        xyz = torch.as_tensor([xyz], device=device)
        rest = []
        if isinstance(axis, int): axis = [axis,]*len(args)
        for i, arg in enumerate(args):
            if type(arg) in [np.ndarray, torch.Tensor]:
                arg = arg.tolist()
            my_ax = axis[i]
            c = [j for j in range(len(np.shape(arg)))]
            c[0] = my_ax
            c[my_ax] = 0
            arg_t = np.transpose(arg, c)
            arg_s = [arg_t[j] for j in self.sortidx]
            rest.append( np.transpose(arg_s,c) )
        return Z, xyz, *rest
    
    def reorder_output(self, output, *args, axis=0):
        """
        Restores ordering of any provided array according to original atom ordering
        
        Parameters
        ----------
        args : ndarray, shape (nAtoms, ...)
            arrays of atomic property obtained from pyseqm calculation
            to be reordered
        axis : int / array-like
            axis along which to reorder arrays
            (int: same for all, array: one axis per array)
                
        Returns
        -------
        new_args : ndarray, shape (nAtoms, ...)
            reordered input array according to original atom ordering

        """
        if isinstance(axis, int): axis = [axis,]*(1+len(args))
        if type(output) is torch.Tensor:
            output = output.detach().numpy()
        c = [j for j in range(len(np.shape(output)))]
        c[0] = axis[0]
        c[axis[0]] = 0
        soutput = np.transpose(output,c)
        soutput = [soutput[j] for j in self.reverse]
        soutput = np.transpose(soutput,c)
        rest = []
        for i, arg in enumerate(args):
            if type(arg) is torch.Tensor: arg = arg.detach().numpy()
            my_ax = axis[i+1]
            c = [j for j in range(len(np.shape(arg)))]
            c[0] = my_ax
            c[my_ax] = 0
            sarg = np.transpose(arg, c)
            sarg = [sarg[j] for j in self.reverse]
            rest.append( np.transpose(sarg,c) )
        return soutput, *rest
        

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
    
def get_default_bounds(parameters, species, method, parameter_dir, 
                       change_zeros=False, with_eq=True, tight_bounds=True):
    """
    Returns default bounds (spanning wide space) and constraints (physically-
    reasonable space). Bounds are given as multiples of default values.
    
    If change_zeros: Extend bounds for entries that have 0. as default value
        (These should be irrelevant for the system and might only complicate,
         for example, optimization, but becomes relevant when extending the
         default parametrization.)
    If with_eq:
        allow for equality constraints defined through equivalent upper and
        lower bounds (otherwise put equality condition in constraints)
    
    Parameters:
    -----------
    parameters : tuple of str
        names of parameters to return original values and bounds for
    species : torch.Tensor / array-like, shape (nAtoms,)
        atomic numbers
    method : str
        type of NDDO calculation to perform [AM1, PM3, MNDO]
    parameter_dir : str
        path to directory with default parameters
    change_zeros : bool (default False)
        whether or not to change parameters with zero default value
    with_eq: bool (default True)
        allow equality constraints in bounds, otherwise return loose
        bounds and tight constraints
    tight_bounds : bool (default True)
        if not using equality constraints, use tight placeholder bounds
        (ABSOLUTELY NOT recommended for use with change_zeros=True)
    
    Returns:
    --------
    pbounds : scipy.optimize Bounds object
        bounds for parameters (parameter space)
    pconstr : scipy LinearConstraint object
        constraints for optimization (parameter space)
    ubounds : scipy.optimize Bounds object
        bounds for parameters in unit hypercube
    uconstr : scipy LinearConstraint object
        constraints for optimization within unit hypercube
    """
    if change_zeros:
        nonsensemsg  = "Changing zero defaults only makes sense with"
        nonsensemsg += "`with_eq=False` and `tight_bounds=False`!"
        if with_eq or (~with_eq and tight_bounds):
            raise ValueError(nonsensemsg)
    nA = species.size()[-1]
    bounds = [default_multi_bound[par] for par in parameters]
    bounds_expanded = np.array([[b,]*nA for b in bounds]).T
    lowupp = np.array([bounds_expanded[0].T.flatten(),
                       bounds_expanded[1].T.flatten()])
    pdef = get_default_parameters(species, method, parameter_dir,parameters)
    def4b = pdef.clone().detach().numpy()
    def4c = def4b.copy().flatten()
    if with_eq:
        def4b = def4b.flatten()
        low_b, upp_b = np.sort(lowupp * def4b, axis=0)
        pbounds = Bounds(low_b, upp_b)
        eq_mask = np.abs(upp_b - low_b) < 1e-12
        low_uc = np.where(eq_mask, 1., 0.)
        ubounds = Bounds(low_uc, np.ones_like(low_uc))
        return pbounds, None, ubounds, None
    else:
        def4c = def4b.copy().flatten()
        zero_idx = np.argwhere(np.abs(def4b)<1e-8)
        ## bounds in some optimizers cannot be equal (fixed parameter)
        ## as this causes problem in mapping parameters to [0;1).
        ## Use placeholder bounds and then constraints to fix values.
        ## The bounds can be tight (+/- eps) or loose (see `max_defaults`).
        if not tight_bounds:
            for ij in zero_idx:
                def4b[tuple(ij)] = max_defaults[method][parameters[ij[0]]]
        def4b = def4b.flatten()
        low_b, upp_b = np.sort(lowupp * def4b, axis=0)
        if tight_bounds:
            eq_mask = np.abs(upp_b - low_b) < 1e-12
            low_b = np.where(eq_mask, low_b - 1e-6, low_b)
            upp_b = np.where(eq_mask, upp_b + 1e-6, upp_b)
        else:
            low_b = np.minimum(low_b, def4c)
            upp_b = np.maximum(upp_b, def4c)
        pbounds = Bounds(low_b, upp_b)
        if change_zeros: def4c = def4b
        low_c, upp_c = np.sort(lowupp * def4c, axis=0)
        pconstr = LinearConstraint(np.eye(low_c.size), low_c, upp_c)
        ## unit cube
        low_ub = scale_to_unit_cube(low_b, pbounds)
        upp_ub = scale_to_unit_cube(upp_b, pbounds)
        ubounds = Bounds(low_ub, upp_ub)
        low_uc = scale_to_unit_cube(low_c, pbounds)
        upp_uc = scale_to_unit_cube(upp_c, pbounds)
        uconstr = LinearConstraint(np.eye(low_uc.size), low_uc, upp_uc)
        return pbounds, pconstr, ubounds, uconstr
    
def get_parameter_names(method):
    """ Returns tuple of all parameter nemaes for <method>. """
    return tuple(parameter_names[method])
    
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
    ordered tuple of input for positional args of `func`
    """
    argspec = getfullargspec(func)
    func_defaults = argspec.defaults
    func_kwargs = argspec.args[-len(func_defaults):]
    ordered_args = []
    for i, key in enumerate(func_kwargs):
        ordered_args.append(kwargs.get(key,func_defaults[i]))
    return tuple(ordered_args)

   
def post_process_result(p, p_init, loss_func, nAtoms, unit_cube=False, 
                        bounds=None, jac=None):
    loss_init = loss_func(p_init)
    if type(loss_init) in [np.ndarray, list]:
        if np.size(loss_init) == 1: loss_init = float(loss_init)
    loss_opt = loss_func(p)
    if type(loss_opt) in [np.ndarray, list]:
        if np.size(loss_opt) == 1: loss_opt = float(loss_opt)
    dloss = loss_opt - loss_init
    ## get gradient wrt input p (on whatever scale p is!)
    if callable(jac):
        gradL = jac(p)
    else:
        gradL = approx_fprime(p, loss_func)
    if unit_cube:
        ## output properties in original space (not for unit cube)
        p_init = scale_from_unit_cube(p_init, bounds)
        p = scale_from_unit_cube(p, bounds)
        if callable(jac):
            ## Jacobian has to be defined for unit cube (i.e., mapping to
            ## unit cube is not part of the gradient - contrary to below)
            gradL = scale_from_unit_cube(gradL, bounds, for_gradient=True)
        else:
            ## if input p is on unit cube scale, `loss_func` contains
            ## an initial `scale_from_unit_cube`, which is reflected
            ## in the FD gradient, but shouldn't enter the final gradient.
            ## Thus, we need to do the *inverse* operation here.
            ## (equivalent to remapping p and `loss_func` and then taking
            ##  the FD gradient)
            gradL = scale_to_unit_cube(gradL, bounds, for_gradient=True)
    p_ref = np.reshape(p_init,(-1,nAtoms))
    p_opt = np.reshape(p,(-1,nAtoms))
    dp = p_opt - p_ref
    gradL = np.reshape(gradL,(-1,nAtoms))
    return p_opt, dp, gradL, loss_opt, dloss
    

class Logger(object):
    from sys import stdout
    def __init__(self, filename=None):
        self.out = stdout if filename is None else open(filename, "a")
    def write(self, message): self.out.write(message)
    def __getattr__(self, attr): return getattr(stdout, attr)
    def flush(self): self.out.flush()

class write_obj:
    def __init__(self, writer, closer):
        self.writer, self.closer = writer, closer
    def __call__(self, s):
        self.writer(s)
    def close(self): self.closer()

def get_writer(writer_in):
    if isinstance(writer_in, write_obj): return writer_in
    from io import IOBase
    if writer_in == 'stdout':
        from sys import stdout
        writer_call = stdout.writelines
        def closer_call(): pass
    elif type(writer_in) is str:
        f = open(writer_in, 'a')
        writer_call = f.write
        closer_call = f.close
    elif isinstance(writer_in, IOBase):
        writer_call = writer_in.write
        closer_call = writer_in.close
    else:
        msg  = "Input 'writer' has to be 'stdout', file_name, or "
        msg += "file object (io.IOBase class)."
        raise RuntimeError(msg)
    writer_out = write_obj(writer_call, closer_call)
    return writer_out
    

def write_param_summary(p, dp, loss_opt, dloss, pname_list, symbols, 
                        writer='stdout', ID='#OPTIMIZED', close_after=False):
    nAtoms = len(symbols)
    writer = get_writer(writer)
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
    if close_after: writer.close()
    return

    
def write_gradient_summary(gradL, symbols, pname_list, writer='stdout', close_after=False):
    writer = get_writer(writer)
    writer('---------------------------------------\n')
    writer('#DLOSS_DPARAM: Gradient of current loss\n')
    sstr  = '                  '
    sstr += ''.join(['  {0:6s}   '.format(s) for s in symbols])
    writer(sstr[:-2]+'\n')
    for i, pname in enumerate(pname_list):
        gstr  = ' {0:<12s}: '.format(pname)
        gstr += ''.join(['{0: 5.2e}  '.format(g) for g in gradL[i]])
        writer(gstr[:-2]+'\n')
    gradL = gradL.flatten()
    i = np.argmax(np.abs(gradL))
    nA = np.size(symbols)
    max_par = pname_list[i//nA]
    max_at  = i%nA
    max_elm = symbols[max_at]
    writer('---------------------------------------\n')
    writer('#MAX_GRAD: Component of loss gradient with max. abs. value\n')
    writer(' {0:<12s} for Atom {1: 3d} ({2:>2s}):'.format(max_par,max_at,max_elm))
    writer('  {0: 5.2e}  \n'.format(gradL[i]))
    if close_after: writer.close()
    return


#--EOF--# 
