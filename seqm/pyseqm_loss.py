import numpy as np
import torch
from functools import lru_cache
from torch.autograd import grad as agrad
from seqm.basics import Parser
from seqm.seqm_functions.constants import Constants
from seqm.pyseqm_helpers import get_ordered_args


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
device = torch.device('cuda') if has_cuda else torch.device('cpu')


@lru_cache(maxsize=16)
def run_calculation(p, calculator=None, coordinates=None,
                    species=None, custom_params=()):
    """
    Returns total energy of SEQC calculation with custom parameters.
    
    Parameters (due to lru_cache, all inputs must be hashable!)
    ----------
    p : torch.Tensor, shape (#custom parameters,)
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
    
    res = calculator(const, coordinates, species, learnedpar, all_terms=True)
    parser = Parser(calculator.seqm_parameters)
    n_occ = parser(const, species, coordinates)[4]
    homo, lumo = n_occ - 1, n_occ
    orb_eigs = res[6][0]
    gap = orb_eigs[lumo] - orb_eigs[homo]
    res = [*res[:-1], gap, res[-1]]
    return p, coordinates, res
    

def clear_results_cache(): run_calculation.cache_clear()


def energy_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Eref=0.):
    """
    Returns squared loss in energy
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    calculator : pyseqm.basics.Energy object
        SE-QC calculator instance for system
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    Eref : float or torch.Tensor(float), shape ()
        reference energy of system in eV
    """
    p, coordinates, res = run_calculation(p, 
                                          calculator=calculator,
                                          coordinates=coordinates, 
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail: return 1e10
    deltaE = E.sum() - Eref
    L = deltaE*deltaE
    return L.detach().numpy()
    
def energy_loss_jac(p, popt_list=[], calculator=None, coordinates=None,
              species=None, Eref=0.):
    """
    Gradient of square loss in energy
    """
    p, coordinates, res = run_calculation(p, 
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail: return 1e10*np.ones_like(p).flatten()
    deltaE = E.sum() - Eref
    dE_dp = agrad(E, p)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = deltaE * dE_dp
    return 2.0 * dL_dp.detach().numpy()
    

def forces_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Fref=None):
    """
    Returns squared loss in atomic forces
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    calculator : pyseqm.basics.Energy object
        SE-QC calculator instance for system
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    Fref : torch.Tensor, shape (nAtoms, 3)
        reference forces (-gradient) of system in eV/Ang
    """
    Fref = torch.as_tensor(Fref, device=device)
    p, coordinates, res = run_calculation(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail: return 1e10
    F = -agrad(E, coordinates)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum()
    return L.detach().numpy()
    
def force_loss_jac(p, popt_list=[], calculator=None,coordinates=None,
                   species=None, Fref=0., *args, **kwargs):
    """
    Gradient of square loss of forces
    
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
    p, coordinates, res = run_calculation(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail: return 1e10*np.ones_like(p).flatten()
    F = -agrad(E, coordinates, create_graph=True)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum()
    dL_dp = agrad(L, p)[0]
    return dL_dp.detach().numpy().flatten()
#    deltaE = E.sum() - Eref
#    dE_dp = agrad(E, p, create_graph=with_forces)[0]
#    dE_dp = dE_dp.flatten()
#    dL_dp = torch.zeros_like(dE_dp)
#    deltaE_dr = agrad(deltaE, coordinates, retain_graph=True)[0]
#    for i, dE_dpi in enumerate(dE_dp):
#        d2E_drdpi = agrad(dE_dpi, coordinates, retain_graph=True)[0]
#        grad_prod = deltaE_dr * d2E_drdpi
#        grad_prod_sum = grad_prod.sum()
#        dLf_dp[i] = dLf_dp[i] + grad_prod_sum
#    return 2.0 * dL_dp.detach().numpy()
    

def gap_loss(p, popt_list=[], calculator=None, coordinates=None,
             species=None, gap_ref=0., *args, **kwargs):
    """
    Returns squared loss in HOMO-LUMO gap
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    calculator : pyseqm.basics.Energy object
        SE-QC calculator instance for system
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    gap_ref : float or torch.Tensor, shape ()
        reference HOMO-LUMO gap in eV
    """
    p, coordinates, res = run_calculation(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    gap, SCFfail = res[9], res[-1]
    if SCFfail: return 1e10
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    return L.detach().numpy()

def gap_loss_jac(p, popt_list=[], calculator=None, coordinates=None,
                 species=None, gap_ref=0., *args, **kwargs):
    """
    Returns squared loss in atomic forces
    
    Parameters:
    -----------
    p : torch.Tensor, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : tuple, len #custom parameters / nAtoms
        names of custom parameters, e.g., ['g_ss', 'zeta_s']
    calculator : pyseqm.basics.Energy object
        SE-QC calculator instance for system
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions in Ang
    species : torch.Tensor, shape (nAtoms,)
        atomic numbers in system
    gap_ref : float or torch.Tensor, shape ()
        reference HOMO-LUMO gap in eV
    """
    p, coordinates, res = run_calculation(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    gap, SCFfail = res[9], res[-1]
    if SCFfail: return 1e10
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    dL_dp = agrad(L, p)[0]
    return dL_dp.detach().numpy().flatten()
    

class LossConstructor:
    def __init__(self, **kwargs):
        self.L = 0.
        self.include = []
        req = ['popt_list', 'calculator', 'coordinates', 'species']
        if any(required not in kwargs for required in req):
            msg = "Please specify'"+"', '".join(req)+"' as kwargs!"
            raise ValueError(msg)
        self.general_kwargs = kwargs
    
    def __call__(self, p, *args, **kwargs):
        p = torch.as_tensor(p, device=device)
        L = 0.
        for prop in self.include:
            L_i = eval(prop+'_loss(p, *self.'+prop+'_args)')
            exec('self.'+prop+'_loss_val = L_i')
            L += eval('self.weight_'+prop+' * L_i')
        clear_results_cache()
        return L
        
    def add_loss(self, prop, weight=1., **kwargs):
        if prop not in ['energy','forces','gap']:
            raise ValueError()
        self.include.append(prop)
        exec('self.weight_'+prop+' = weight')
        kwargs.update(self.general_kwargs)
        exec('self.'+prop+'_args = get_ordered_args('+prop+'_loss, **kwargs)')
        
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
        

