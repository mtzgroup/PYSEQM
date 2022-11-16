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
    
    try:  # calculation might fail for random choice of parameters
        res = calculator(const, coordinates, species, learnedpar, all_terms=True)
        parser = Parser(calculator.seqm_parameters)
        n_occ = parser(const, species, coordinates)[4]
        homo, lumo = n_occ - 1, n_occ
        orb_eigs = res[6][0]
        gap = orb_eigs[lumo] - orb_eigs[homo]
        res = [*res[:-1], gap, res[-1]]
    except RuntimeError:
        res = [None,]*11
        res[-1] = True
    return p, coordinates, res
    

def clear_results_cache(): run_calculation.cache_clear()


def energy_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Eref=0.):
    """
    Returns squared loss in energy per atom
    
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
    if SCFfail: return np.inf
    deltaE = E.sum() - Eref
    L = deltaE*deltaE / species.shape[0]
    return L.detach().numpy()
    
def energy_loss_jac(p, popt_list=[], calculator=None, coordinates=None,
              species=None, Eref=0.):
    """
    Gradient of square loss in energy per atom
    """
    p, coordinates, res = run_calculation(p, 
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail:
        dummy = p.clone().detach()
        return np.inf*np.sign(dummy).flatten()
    deltaE = E.sum() - Eref
    dE_dp = agrad(E, p, retain_graph=True)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = deltaE * dE_dp / species.shape[0]
    return 2.0 * dL_dp.detach().numpy()
    

def forces_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Fref=None):
    """
    Returns squared loss in atomic forces per atom
    
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
    if SCFfail: return np.inf
    F = -agrad(E, coordinates)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum() / species.shape[0]
    return L.detach().numpy()
    
def forces_loss_jac(p, popt_list=[], calculator=None,coordinates=None,
                    species=None, Fref=0., *args, **kwargs):
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
    p, coordinates, res = run_calculation(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[-1]
    if SCFfail:
        dummy = p.clone().detach()
        return np.inf*np.sign(dummy).flatten()
    F = -agrad(E, coordinates, create_graph=True)[0][0]
    F = F - torch.sum(F, dim=0)  # remove COM force
    L = torch.square(F - Fref).sum()
    dL_dp = agrad(L, p, retain_graph=True)[0] / species.shape[0]
    return dL_dp.detach().numpy().flatten()
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
    if SCFfail: return np.inf
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
    if SCFfail:
        dummy = p.clone().detach()
        return np.inf*np.sign(dummy).flatten()
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    dL_dp = agrad(L, p, retain_graph=True)[0]
    return dL_dp.detach().numpy().flatten()
    

class LossConstructor:
    def __init__(self, **kwargs):
        self.include = []
        self.implemented_properties = ['energy','forces','gap']
        req = ['popt_list', 'calculator', 'coordinates', 'species']
        if any(required not in kwargs for required in req):
            msg = "Please specify'"+"', '".join(req)+"' as kwargs!"
            raise ValueError(msg)
        self.general_kwargs = kwargs
    
    def __call__(self, p, *args, **kwargs):
        p = torch.as_tensor(p, device=device)
        self.L = 0.
        clear_results_cache()
        for prop in self.include:
            L_i = eval(prop+'_loss(p, *self.'+prop+'_args)')
            exec('self.'+prop+'_loss_val = L_i')
            self.L += eval('self.weight_'+prop+' * L_i')
        clear_results_cache()
        return self.L
    
    def loss_and_jac(self, p, *args, **kwargs):
        p = torch.as_tensor(p, device=device)
        self.L, self.dLdp = 0., np.zeros_like(p)
        clear_results_cache()
        for prop in self.include:
            L_i = eval(prop+'_loss(p, *self.'+prop+'_args)')
            dLdp_i = eval(prop+'_loss_jac(p, *self.'+prop+'_args)')
            exec('self.'+prop+'_loss_val = L_i')
            exec('self.'+prop+'_loss_grad = dLdp_i')
            self.L += eval('self.weight_'+prop+' * L_i')
            self.dLdp += eval('self.weight_'+prop+' * dLdp_i')
        clear_results_cache()
        return (self.L, self.dLdp)
    
    def jac(self, p, *args, **kwargs):
        p = torch.as_tensor(p, device=device)
        self.dLdp = np.zeros_like(p)
        clear_results_cache()
        for prop in self.include:
            dLdp_i = eval(prop+'_loss_jac(p, *self.'+prop+'_args)')
            exec('self.'+prop+'_loss_grad = dLdp_i')
            self.dLdp += eval('self.weight_'+prop+' * dLdp_i')
        clear_results_cache()
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
