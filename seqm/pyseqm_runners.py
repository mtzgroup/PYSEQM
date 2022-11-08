import numpy as np
import torch
from functools import lru_cache
from torch.autograd import grad as agrad
from seqm.seqm_functions.constants import Constants


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
device = torch.device('cuda') if has_cuda else torch.device('cpu')


@lru_cache(maxsize=156, typed=False)
def run_custom_seqc(p, calculator=None, coordinates=None,
                    species=None, custom_params=[]):
    """
    Returns total energy of SEQC calculation with custom parameters.
    
    Parameters
    ----------
    p : Tensor, shape (number of custom parameters, number of atoms)
        array of custom parameters
    calculator : pyseqm Energy calculator object
        list of names of custom parameters as used in parameter file
    coordinates : Tensor, shape (nAtoms, 3)
        atomic positions in Angstrom
    species : Tensor, shape (nAtoms,)
        atomic numbers in descending order (coordinates accordingly)
    custom_params : list
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
        nconv : bool
            whether SCF convergence of SE-QC calculation failed or not
    
    """
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    const = Constants().to(device)
    coordinates.requires_grad_(True)
    p.requires_grad_(True)
    learnedpar = {pname:p[i] for i, pname in enumerate(custom_params)}
    
    res = calculator(const, coordinates, species, learnedpar, all_terms=True)
    return p, coordinates, res
    

def energy_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Eref=0., *args, **kwargs):
    """
    If with_forces=False(default): Returns squared loss in atomization energy
    If with_forces=True: Returns weightE*Loss(Energy) + weightF*Loss(Forces)
    
    Parameters:
    -----------
    p : ndarray, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : ndarray(-like), shape (#custom parameters / nAtoms,)
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
    p, coordinates, res = run_custom_seqc(p, 
                                          calculator=calculator,
                                          coordinates=coordinates, 
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = seqm_result[1], seqm_result[9]
    if SCFfail: return 1e10
    deltaE = E.sum() - Eref
    L = deltaE*deltaE
    return L.detach().numpy()
    
def energy_loss_jac(p, popt_list=[], calculator=None, coordinates=None,
              species=None, Eref=0., *args, **kwargs):
    """
    Gradient of square loss in energy
    """
    p, coordinates, res = run_custom_seqc(p, 
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    if SCFfail: return 1e10*np.ones_like(p).flatten()
    deltaE = E.sum() - Eref
    dE_dp = agrad(E, p)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = deltaE * dE_dp
    return 2.0 * dL_dp.detach().numpy()
    

def force_loss(p, popt_list=[], calculator=None, coordinates=None,
                species=None, Fref=None, *args, **kwargs):
    """
    Returns squared loss in atomic forces
    
    Parameters:
    -----------
    p : ndarray, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : ndarray(-like), shape (#custom parameters / nAtoms,)
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
    if Fref is None:
        Fref = torch.zeros_like(coordinates, device=device)
    Fref = torch.as_tensor(Fref, device=device)
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    coordinates.requires_grad_(True)
    p, coordinates, res = run_custom_seqc(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[9]
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
    p, coordinates, res = run_custom_seqc(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    E, SCFfail = res[1], res[9]
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
    p : ndarray, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : ndarray(-like), shape (#custom parameters / nAtoms,)
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
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    p, coordinates, res = run_custom_seqc(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    orb_enes, SCFfail = res[6], res[9]
    if SCFfail: return 1e10
    ## get gap from orb_enes!
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    return L.detach().numpy()

def gap_loss(p, popt_list=[], calculator=None, coordinates=None,
             species=None, gap_ref=0., *args, **kwargs):
    """
    Returns squared loss in atomic forces
    
    Parameters:
    -----------
    p : ndarray, shape (#custom parameters,)
        Custom parameters in the format
            ([[first custom parameter for all atoms],
              [second custom parameter for all atoms],
              ..., ]).flatten()
    popt_list : ndarray(-like), shape (#custom parameters / nAtoms,)
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
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    p, coordinates, res = run_custom_seqc(p,
                                          calculator=calculator,
                                          coordinates=coordinates,
                                          species=species,
                                          custom_params=popt_list)
    orb_enes, SCFfail = res[6], res[9]
    if SCFfail: return 1e10
    ## get gap from orb_enes!
    deltaG = gap - gap_ref
    L = deltaG * deltaG
    dL_dp = agrad(L, p)[0]
    return dL_dp.detach().numpy().flatten()
    

