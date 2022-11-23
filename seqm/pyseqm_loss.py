import numpy as np
import torch
from functools import lru_cache
from torch.autograd import grad as agrad
from seqm.basics import Parser, Energy
from seqm.seqm_functions.constants import Constants
from seqm.pyseqm_helpers import get_ordered_args, scale_from_unit_cube, \
                                scale_to_unit_cube

MAX_CACHE_SIZE=0
LFAIL = torch.tensor(torch.inf)


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device('cuda')
    sp2_def = [True, 1e-5]
else:
    device = torch.device('cpu')
    sp2_def = [False]


class PyseqmContainer:
    """ Class holding result of seqm calculation. """
    def __init__(self):
        self.seqm_result = None
        self.available_properties = {'atomization_energy':0, 'scf_energy':1,
                            'electronic_energy':2, 'nuclear_energy':3, 
                            'isolated_energies':4, 'nuclear_energy_pairs':5,
                            'orbital_energies':6, 'density_matrix':7, 
                            'charges':8, 'gap':9}
    
    def update_result(self, p, calculator, coordinates, species, 
                      custom_params):
        """ Perform new calculation and store settings & result. """
        self.p = p.reshape((len(custom_params),-1))
        self.p.requires_grad_(True)
        self.learnedpar = {n:self.p[i] for i,n in enumerate(custom_params)}
        self.calculator = calculator
        self.coordinates = coordinates
        self.coordinates.requires_grad_(True)
        self.species = species
        self.custom_params = custom_params
        self.const = Constants().to(device)
        self.seqm_result = self.run_calculation()
        self.p = self.p.flatten()
        
    def get(self, prop, p, calculator=None, coordinates=None, 
            species=None, custom_params=()):
        """
        Returns total energy of SEQC calculation with custom parameters.
        
        Parameters
        ----------
        prop : str
            one of atomization_energy, scf_energy, electronic_energy, 
            nuclear_energy, isolated_energies, nuclear_energy_pairs, 
            orbital_energies, density_matrix, charges, or gap
        p : torch.Tensor, shape (#custom parameters*nAtoms)
            custom parameters
        calculator : pyseqm Energy calculator object
            list of names of custom parameters as used in parameter file
        coordinates : Tensor, shape (nAtoms, 3)
            atomic positions in Angstrom
        species : Tensor, shape (nAtoms,)
            atomic numbers in descending order (coordinates accordingly)
        custom_params : tuple
            names of custom parameters defined in p
        
        Returns
        -------
        p : torch.Tensor, shape (#custom parameters * nAtoms)
            custom parameters (after inclusion in computational graph)
        coordinates : torch.Tensor, shape (nAtoms, 3)
            atomic positions (after inclusion in computational graph)
        <prop> : float/tensor/object
            value/instance of <prop> requested in input
        fail : bool
            flag if SE-QC calculation failed
        """
        prop = prop.lower()
        if prop not in self.available_properties:
            raise ValueError("property '"+prop+"' not implemented.")
        need_calc = self.check4changes(p=p, calculator=calculator, 
                                       coordinates=coordinates, 
                                       species=species,
                                       custom_params=custom_params)
        if need_calc:
            self.update_result(p, calculator, coordinates, species,
                               custom_params)
        out = self.seqm_result[self.available_properties[prop]]
        fail = self.seqm_result[-1]
        return self.p.clone(), self.coordinates.clone(), out.clone(), fail
        
    def check4changes(self, p, calculator, coordinates, species, 
                      custom_params):
        if self.seqm_result is None: return True
        for a in ['p','calculator','coordinates','species','custom_params']:
            if not hasattr(self, a): return True
        ## arrays (float or int)
        if (p-self.p).abs().sum() > 1e-10: return True
        if (coordinates-self.coordinates).abs().sum() > 1e-10: return True
        if (species-self.species).abs().sum() > 0: return True
        ## tuple
        if custom_params != self.custom_params: return True
        ## seqm "calculator" (see __eq__ in basics.Energy and basics.Force)
        if calculator != self.calculator: return True
        return False
        
    def run_calculation(self):
        """ Run SEQC calculation with custom parameters. """
        try:  # calculation might fail for random choice of parameters
            res = self.calculator(self.const, self.coordinates, self.species, 
                                  self.learnedpar, all_terms=True)
            parser = Parser(self.calculator.seqm_parameters)
            n_occ = parser(self.const, self.species, self.coordinates)[4]
            homo, lumo = n_occ - 1, n_occ
            orb_eigs = res[6][0]
            gap = orb_eigs[lumo] - orb_eigs[homo]
            res = [*res[:-1], gap, res[-1]]
        except RuntimeError:
            res = torch.tensor([torch.nan,]*11)
            res[-1] = True
        return res
    
    def clear_results(self):
        self.seqm_result = None
    

@lru_cache(maxsize=MAX_CACHE_SIZE)
def run_calculation(p, coordinates=None, species=None, custom_params=(),
                    setting_keys=(), setting_vals=()):
    """
    Returns total energy of SEQC calculation with custom parameters.
    
    Parameters (due to lru_cache, all inputs must be hashable!)
    ----------
    p : torch.Tensor, shape (#custom parameters,)
        custom parameters
    coordinates : Tensor, shape (nAtoms, 3)
        atomic positions in Angstrom
    species : Tensor, shape (nAtoms,)
        atomic numbers in descending order (coordinates accordingly)
    setting_keys : tuple
        keyword options for seqm Energy calculator
    setting_vals : tuple
        values corresponding to keyword options of seqm Energy calculator
    custom_params : tuple
        names of custom parameters defined in p
    
    Returns
    -------
    p : torch.Tensor, shape (#custom parameters, nAtoms)
        custom parameters (after inclusion in computational graph)
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions (after inclusion in computational graph)
    res : tuple
        Tuple containing results of custom SE-QC calculation:
        Eat : torch.Tensor, shape (1,)
            atomization energy [eV?], if Hf_flag [kcal/mol?]
        E : torch.Tensor, shape (1,)
            total SCF energy of system [eV]
        Eel : torch.Tensor, shape (1,)
            electronic energy [eV]
        Enuc : torch.Tensor, shape (1,)
            nuclear ("repulsive") energy [eV]
        Eiso : torch.Tensor, shape (1,)
            sum of energies of isolated atoms [eV]
        EnucAB : torch.Tensor, shape (N, N)
            individual nuclear ("repulsive") energies for all pairs [eV]
        orb_eigs : torch.Tensor, shape (Nelec,)
            orbital energies [eV?]
        P : torch.Tensor, shape (nOrbs, nOrbs)
            density matrix
        q : torch.Tensor, shape (nAtoms,)
            atomic charges (Mulliken?) [e]
        gap : torch.Tensor, shape () / float
            HOMO-LUMO gap [eV]
        nconv : bool
            whether SCF convergence of SE-QC calculation failed or not
    
    """
    p = p.reshape((len(custom_params),-1))
    const = Constants().to(device)
    coordinates.requires_grad_(True)
    p.requires_grad_(True)
    learnedpar = {pname:p[i] for i, pname in enumerate(custom_params)}
    settings = dict(zip(setting_keys, setting_vals))
    calculator = Energy(settings).to(device)
    
    try:  # calculation might fail for random choice of parameters
            # Eat, E, Eel, Enuc, Eiso, EnucAB, orb_eigs, P, q, gap, fail
        res = calculator(const, coordinates, species, learnedpar, all_terms=True)
        parser = Parser(calculator.seqm_parameters)
        n_occ = parser(const, species, coordinates)[4]
        homo, lumo = n_occ - 1, n_occ
        orb_eigs = res[6][0]
        gap = orb_eigs[lumo] - orb_eigs[homo]
        res = [*res[:-1], gap, res[-1]]
    except RuntimeError:
        res = torch.tensor([torch.nan,]*11)
        res[-1] = True
    return p, coordinates, res
    
def clear_results_cache(): run_calculation.cache_clear()


def energy_loss(p, popt_list=(), coordinates=None, species=None, 
                seqm_settings={}, Eref=0.):#, cache_container=None):
    """
    Returns squared loss in energy per atom and energy
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    seqm_settings : dict
        dictionary for seqm Energy calculator
    Eref : float or torch.Tensor(float), shape ()
        reference energy of system in eV
#    cache_container : PyseqmContainer object (or None)
#        if we want to use caching from previous calculation,
#        pass the PyseqmContainer object that caches results
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, E, SCFfail = cache_container.get('scf_energy', p, 
#                                          calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    E, SCFfail = res[1], res[-1]
    if SCFfail: return LFAIL, LFAIL
    deltaE = E - Eref
    L = deltaE*deltaE / species.shape[0]
    return L, E
    
def energy_loss_jac(p, popt_list=(), coordinates=None, species=None, 
                    seqm_settings={}, Eref=0.):#, cache_container=None):
    """
    Gradient of square loss in energy per atom
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, E, SCFfail = cache_container.get('scf_energy', p,
#                                          calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    E, SCFfail = res[1], res[-1]
    if SCFfail:
        dummy = p.clone().detach()
        return torch.inf*torch.sign(dummy).flatten()
    deltaE = E - Eref
    dE_dp = agrad(E, p, retain_graph=True)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = deltaE * dE_dp / species.shape[0]
    return 2.0 * dL_dp
    

def atomization_loss(p, popt_list=(), coordinates=None, species=None, 
                     seqm_settings={}, Eref=0.):#, cache_container=None):
    """
    Returns squared loss in atomization energy per atom and 
    atomization energy
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    seqm_settings : dict
        dictionary for seqm Energy calculator
    Eref : float or torch.Tensor(float), shape ()
        reference atomization energy of system in eV
#    cache_container : PyseqmContainer object (or None)
#        if we want to use caching from previous calculation,
#        pass the PyseqmContainer object that caches results
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, Eat, SCFfail = cache_container.get('atomization_energy',
#                                       p, calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    Eat, SCFfail = res[0], res[-1]
    if SCFfail: return LFAIL, LFAIL
    deltaE = Eat - Eref
    L = deltaE*deltaE / species.shape[0]
    return L, Eat

def atomization_loss_jac(p, popt_list=(), coordinates=None, species=None, 
                         seqm_settings={}, Eref=0.):#, cache_container=None):
    """
    Gradient of square loss in atomization energy per atom
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, Eat, SCFfail = cache_container.get('atomization_energy', 
#                                       p, calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    Eat, SCFfail = res[0], res[-1]
    if SCFfail:
        dummy = p.clone().detach()
        return torch.inf*torch.sign(dummy).flatten()
    deltaE = Eat - Eref
    dE_dp = agrad(Eat, p, retain_graph=True)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = deltaE * dE_dp / species.shape[0]
    return 2.0 * dL_dp
    
def forces_loss(p, popt_list=(), coordinates=None, species=None, 
                seqm_settings={}, Fref=None):#, cache_container=None):
    """
    Returns squared loss in atomic forces per atom and force
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    seqm_settings : dict
        dictionary for seqm Energy calculator
    Fref : torch.Tensor, shape (nAtoms, 3)
        reference forces (-gradient) of system in eV/Ang
#    cache_container : PyseqmContainer object (or None)
#        if we want to use caching from previous calculation,
#        pass the PyseqmContainer object that caches results
    """
    Fref = torch.as_tensor(Fref, device=device)
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, E, SCFfail = cache_container.get('scf_energy', 
#                                       p, calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    E, SCFfail = res[1], res[-1] 
    if SCFfail:
        FFAIL = torch.inf * torch.ones_like(coordinates)
        return LFAIL, FFAIL
    F = -agrad(E, coordinates, retain_graph=True)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum() / species.shape[0]
    return L, F
    
def forces_loss_jac(p, popt_list=(), coordinates=None, species=None, 
                    seqm_settings={}, Fref=0.):#, cache_container=None):
    """
    Gradient of square loss of forces per atom
    
    #Alternative implementation:
    #Assuming Fref = -dEref/dR,
    #    dL[Force]/dp = d [ sum_{ij} (F_{ij} - Fref_{ij})^2 ] / dp
    #          = 2 * [ sum_{ij} d(E-Eref)/dR * d^2E/dRdp ]
    #
    #NOTE: THIS YIELDS EXACTLY THE SAME RESULTS AS DIRECT autograd(L,p)
    #AND THUS ALSO DISAGREES WITH scipy's '2/3-point' NUMERICAL SCHEME
    #(THE PROBLEM APPEARS TO BE IN d^2E / dRdp, ALMOST CERTAINLY BECAUSE
    # OF LACK OR INSTABILITY IN BACKPROP THROUGH THE SCF CYCLE!)
    """
    Fref = torch.as_tensor(Fref, device=device)
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, E, SCFfail = cache_container.get('scf_energy',          
#                                       p, calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    E, SCFfail = res[1], res[-1] 
    if SCFfail:
        dummy = p.clone().detach()
        return torch.inf*torch.sign(dummy).flatten()
    F = -agrad(E, coordinates, create_graph=True)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum()
    dL_dp = agrad(L, p, retain_graph=True)[0] / species.shape[0]
    return dL_dp.flatten()
#    deltaE = E.sum() - Eref
#    dE_dp = agrad(E, p, create_graph=True)[0]
#    dE_dp = dE_dp.flatten()
#    dL_dp = torch.zeros_like(dE_dp)
#    deltaE_dr = agrad(deltaE, coordinates, retain_graph=True)[0]
#    for i, dE_dpi in enumerate(dE_dp):
#        d2E_drdpi = agrad(dE_dpi, coordinates, retain_graph=True)[0]
#        grad_prod = deltaE_dr * d2E_drdpi
#        grad_prod_sum = grad_prod.sum()
#        dLf_dp[i] = dLf_dp[i] + grad_prod_sum
#    return 2.0 * dL_dp.detach().numpy()
    

def gap_loss(p, popt_list=(), coordinates=None, species=None, 
             seqm_settings={}, gap_ref=0.):#, cache_container=None):
    """
    Returns squared loss in HOMO-LUMO gap and HOMO-LUMO gap
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    seqm_settings : dict
        dictionary for seqm Energy calculator
    gap_ref : float or torch.Tensor, shape ()
        reference HOMO-LUMO gap in eV
#    cache_container : PyseqmContainer object (or None)
#        if we want to use caching from previous calculation,
#        pass the PyseqmContainer object that caches results
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, gap, SCFfail = cache_container.get('gap', p,
#                                          calculator    = calculator,
#                                          coordinates   = coordinates,
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    gap, SCFfail = res[9], res[-1]
    if SCFfail: return LFAIL, LFAIL
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    return L, gap

def gap_loss_jac(p, popt_list=(), coordinates=None, species=None, 
                 seqm_settings={}, gap_ref=0.):#, cache_container=None):
    """
    Gradient of squared loss in HOMO-LUMO gap
    """
#    if not isinstance(cache_container, PyseqmContainer):
#        cache_container = PyseqmContainer()
#    p, coordinates, gap, SCFfail = cache_container.get('gap', p,
#                                          calculator    = calculator,
#                                          coordinates   = coordinates, 
#                                          species       = species,
#                                          custom_params = popt_list)
    seqm_keys = tuple(seqm_settings.keys())
    seqm_vals = tuple(seqm_settings.values())
    p, coordinates, res = run_calculation(p,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list,
                                          setting_keys=seqm_keys,
                                          setting_vals=seqm_vals)
    gap, SCFfail = res[9], res[-1]
    if SCFfail:
        dummy = p.clone().detach()
        return torch.inf*torch.sign(dummy).flatten()
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    with torch.autograd.set_detect_anomaly(True):
        dL_dp = agrad(L, p, retain_graph=True)[0]
    return dL_dp.flatten()
    

class LossConstructor:
    def __init__(self, popt_list=(), coordinates=None, species=None, 
                 bounds=None, unit_cube=False, **kwargs):
        self.include = []
        self.implemented_properties = ['energy', 'forces', 'gap',
                                       'atomization']
#        self.cacher = PyseqmContainer()
#        kwargs.update({'cache_container':self.cacher})
        popt_list = tuple(popt_list)
        if unit_cube and bounds is None:
            msg  = "For remapping to the unit hypercube, you need to "
            msg += "specify bounds, my friend!"
            raise ValueError(msg)
        self.bounds = bounds
        self.unit_cube = unit_cube
        elements = [0]+sorted(set(species.reshape(-1).tolist()))
        seqm_settings = {
                         'method'             : 'AM1',
                         'scf_eps'            : 1.0e-6,
                         'scf_converger'      : [2,0.0],
                         'sp2'                : sp2_def,
                         'pair_outer_cutoff'  : 1.0e10,
                         'Hf_flag'            : False,
                        }
        seqm_settings['elements'] = torch.tensor(elements)
        seqm_settings['learned'] = popt_list
        seqm_settings['eig'] = True
        calc_dict = kwargs.pop('calculator_options', {})
        for k,v in calc_dict.items():
            if type(v) is list: calc_dict[k] = tuple(v)
        seqm_settings.update(calc_dict)
        kwargs.update({'seqm_settings':seqm_settings})
        kwargs.update({'popt_list':popt_list,'coordinates':coordinates,
                       'species':species})
        self.general_kwargs = kwargs
    
    def __call__(self, p, *args, **kwargs):
        if self.unit_cube: p = scale_from_unit_cube(p, self.bounds)
        p = torch.as_tensor(p, device=device)
        self.L = 0.
        for prop in self.include:
            L_i, p_i = eval(prop+'_loss(p, *self.'+prop+'_args)')
            L_i = L_i.detach().numpy()
            exec('self.'+prop+'_loss_val = L_i')
            self.L += eval('self.weight_'+prop+' * L_i')
            exec('self.'+prop+'_val = p_i.detach().numpy()')
        return self.L
    
    def loss_and_jac(self, p, *args, **kwargs):
        if self.unit_cube: p = scale_from_unit_cube(p, self.bounds)
        p = torch.as_tensor(p, device=device)
        self.L, self.dLdp = 0., np.zeros_like(p)
        for prop in self.include:
            L_i, p_i = eval(prop+'_loss(p, *self.'+prop+'_args)')
            L_i = L_i.detach().numpy()
            exec('self.'+prop+'_loss_val = L_i')
            self.L += eval('self.weight_'+prop+' * L_i')
            exec('self.'+prop+'_val = p_i.detach().numpy()')
            dLdp_i = eval(prop+'_loss_jac(p, *self.'+prop+'_args)')
            dLdp_i = dLdp_i.detach().numpy()
            exec('self.'+prop+'_loss_grad = dLdp_i')
            self.dLdp += eval('self.weight_'+prop+' * dLdp_i')
        if self.unit_cube: self.dLdp = scale_to_unit_cube(self.dLdp, 
                                        self.bounds, for_gradient=True)
        return (self.L, self.dLdp)
    
    def jac(self, p, *args, **kwargs):
        if self.unit_cube: p = scale_from_unit_cube(p, self.bounds)
        p = torch.as_tensor(p, device=device)
        self.dLdp = np.zeros_like(p)
        for prop in self.include:
            dLdp_i = eval(prop+'_loss_jac(p, *self.'+prop+'_args)')
            dLdp_i = dLdp_i.detach().numpy()
            exec('self.'+prop+'_loss_grad = dLdp_i')
            self.dLdp += eval('self.weight_'+prop+' * dLdp_i')
        if self.unit_cube: self.dLdp = scale_to_unit_cube(self.dLdp, 
                                        self.bounds, for_gradient=True)
        return self.dLdp
    
    def add_loss(self, prop, weight=1., **kwargs):
        """
        Add individual loss evaluators as defined above to loss function.
        If implementing a new property, please add loss functon
        `<property>_loss(...)` above and update self.implemented_properties
        """
        if prop not in self.implemented_properties:
            msg  = "Only '"+"', '".join(self.implemented_properties)
            msg += "' implemented for loss. Check for typos or write "
            msg += "coresponding loss function for '"+prop+"'."
            raise ValueError(msg)
        self.include.append(prop)
        exec('self.weight_'+prop+' = weight')
        kwargs.update(self.general_kwargs)
        exec('self.'+prop+'_args = get_ordered_args('+prop+'_loss, **kwargs)')
        
    def get_loss(self):
        """ Return current loss """
        if not hasattr(self, 'L'): raise ValueError("Loss not calculated yet")
        return self.L
    
    def get_loss_jac(self):
        """ return gradient of current loss """
        if not hasattr(self, 'dLdp'):
            raise ValueError("Loss gradient not calculated yet")
        return self.dLdp
    
    def get_individual_loss(self, prop, include_weight=False):
        """
        Return SQUARED loss of property as included in total loss function
        
        Parameters:
        -----------
        prop : str
            property to return loss value for ('energy', 'forces', ...)
        include_weight : bool (optional, default: False)
            Whether or not to scale squared loss with corresponding weight
            (i.e., as it enters the total loss)
        """
        if not hasattr(self, prop+'_loss_val'):
            raise ValueError("Loss for property '"+prop+"' not available.")
        if include_weight:
            out = eval('self.'+prop+'_loss_val * self.weight_'+prop)
        else:
            out = eval('self.'+prop+'_loss_val')
        if type(out) in [np.ndarray, list]:
            if np.size(out) == 1: out = float(out)
        return out
        
    def get_individual_loss_jac(self, prop, include_weight=False):
        """
        Return gradient of squared loss of property
        
        Parameters:
        -----------
        prop : str
            property to return loss value for ('energy', 'forces', ...)
        include_weight : bool (optional, default: False)
            Whether or not to scale squared loss with corresponding weight
            (i.e., as it enters the total loss gradient)
        """
        if not hasattr(self, prop+'_loss_grad'):
            raise ValueError("Loss gradient for property '"+prop+"' not available.")
        if include_weight:
            out = eval('self.'+prop+'_loss_grad * self.weight_'+prop)
        else:
            out = eval('self.'+prop+'_loss_grad')
        if type(out) in [np.ndarray, list]:
            if np.size(out) == 1: out = float(out)
        return out
    
    def get_property(self, prop):
        """
        Return property as entering loss function
        
        Parameters:
        -----------
        prop : str
            property to return ('energy', 'forces', ...)
        """
        if not hasattr(self, prop+'_val'):
            raise ValueError("Property '"+prop+"' not available.")
        out = eval('self.'+prop+'_val')
        if type(out) in [np.ndarray, list]:
            if np.size(out) == 1: out = float(out)
        return out

