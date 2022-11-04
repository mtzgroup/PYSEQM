import numpy as np
import torch
from torch.autograd import grad as agrad
from seqm.seqm_functions.constants import Constants


torch.set_default_dtype(torch.float64)
has_cuda = torch.cuda.is_available()
device = torch.device('cuda') if has_cuda else torch.device('cpu')


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
    E : torch.Tensor, shape (1,)
        total SCF energy of system (differentiable instance)
    Eat : torch.Tensor, shape (1,)
        atomization energy (differentiable instance)
    q : torch.Tensor, shape (nAtoms,)
        atomic charges (Mulliken)?
    p : torch.Tensor, shape (#custom parameters, nAtoms)
        custom parameters (differentiable instance)
    coordinates : torch.Tensor, shape (nAtoms, 3)
        atomic positions (differentiable instance)
    nconv : bool
        whether SCF convergence of SE-QC calculation failed or not
    
    """
    const = Constants().to(device)
    coordinates.requires_grad_(True)
    p.requires_grad_(True)
    learnedpar = {pname:p[i] for i, pname in enumerate(custom_params)}
    
    res = calculator(const, coordinates, species, learnedpar, all_terms=True)
    Eat, E, q, nconv = res[0], res[1], res[8], res[9]
    return E, Eat, q, p, coordinates, nconv
    

def atomization_loss(p, popt_list=[], calculator=None, coordinates=None,
                     species=None, Eref=0., Fref=None, with_forces=False,
                     weightE=1., weightF=0.):
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
        reference atomization energy of system in eV
    """
    print("Current parameters: ",p.flatten())
    if with_forces:
        Fref = torch.as_tensor(Fref, device=device)
        sum_w = weightE + weightF
        weightE, weightF = weightE/sum_w, weightF/sum_w
    else:
        weightE = 1.
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    coordinates.requires_grad_(with_forces)
    E, Eat, q, p, coordinates, SCFfail = run_custom_seqc(p, 
                                calculator=calculator,
                                coordinates=coordinates, species=species,
                                custom_params=popt_list)
    if SCFfail: return 1e10
    deltaE = E.sum() - Eref
    L = weightE * deltaE*deltaE
    if with_forces:
        F = -agrad(E, coordinates)[0][0]
#        F = F - torch.sum(F, dim=0)  # remove COM force
        deltaF2 = torch.square(F - Fref).sum()
        Lf = weightF * deltaF2
        L = L + Lf
    return L.detach().numpy()
    
def atomization_loss_jac(p, popt_list=[], calculator=None,coordinates=None,
              species=None, Eref=0., Fref=None, with_forces=False,
              weightE=1., weightF=0.):
    """
    Alternative implementation of square loss:
    Assuming Fref = -dEref/dR,
        dL/dp = d [ weightE * (E - Eref)^2 + 
                    weightF * sum_{ij} (F_{ij} - Fref_{ij})^2 ] / dp
              = 2 * [ weightE * (E - Eref) * dE/dp +
                      weightF * sum_{ij} d(E-Eref)/dR * d^2E/dRdp ]
    
    NOTE: THIS YIELDS EXACTLY THE SAME RESULTS AS DIRECT autograd(L,p)
    AND THUS ALSO DISAGREES WITH scipy's '2/3-point' NUMERICAL SCHEME
    (THE PROBLEM APPEARS TO BE IN d^2E / dRdp, ALMOST CERTAINLY BECAUSE
     OF LACK OR INSTABILITY IN BACKPROP THROUGH THE SCF CYCLE!)
    """
    if with_forces:
        Fref = torch.as_tensor(Fref, device=device)
        sum_w = weightE + weightF
        weightE, weightF = weightE/sum_w, weightF/sum_w
    else:
        weightE = 1. 
    if not type(p) is torch.Tensor:
        p = torch.tensor(p.tolist(), device=device, requires_grad=True)
    p = p.reshape((len(popt_list),-1))
    if not type(coordinates) is torch.Tensor:
        coordinates = torch.as_tensor(coordinates.tolist(), device=device)
    coordinates.requires_grad_(with_forces)
    E, Eat, q, p, coordinates, SCFfail = run_custom_seqc(p, 
                                calculator=calculator,
                                coordinates=coordinates, species=species,
                                custom_params=popt_list)
    if SCFfail: return 1e10*np.ones_like(p).flatten()
    deltaE = E.sum() - Eref
    dE_dp = agrad(E, p, create_graph=with_forces)[0]
    dE_dp = dE_dp.flatten()
    dL_dp = weightE * deltaE * dE_dp
    if with_forces:
        dLf_dp = torch.zeros_like(dE_dp)
        deltaE_dr = agrad(deltaE, coordinates, retain_graph=True)[0]
        for i, dE_dpi in enumerate(dE_dp):
            d2E_drdpi = agrad(dE_dpi, coordinates, retain_graph=True)[0]
            grad_prod = deltaE_dr * d2E_drdpi
            grad_prod_sum = grad_prod.sum()
            dLf_dp[i] = dLf_dp[i] + grad_prod_sum
        dL_dp = dL_dp + weightF * dLf_dp
    return 2.0 * dL_dp.detach().numpy()

