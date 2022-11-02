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
    
    """
    const = Constants().to(device)
    coordinates.requires_grad_(True)
    p.requires_grad_(True)
    learnedpar = {pname:p[i] for i, pname in enumerate(custom_params)}
    
    res = calculator(const, coordinates, species, learnedpar, all_terms=True)
    Eat, E, q, nconv = res[0], res[1], res[8], res[9]
    return E, Eat, q, p, coordinates
    

def atomization_loss(p, popt_list=[], calculator=None, coordinates=None,
                     species=None, Eref=0., Fref=None, with_forces=False,
                     weightE=1., weightF=0., jac=False):
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
    make_graph = False
    if with_forces:
        Fref = torch.as_tensor(Fref, device=device)
        sum_w = weightE + weightF
        weightE, weightF = weightE/sum_w, weightF/sum_w
        if jac:
            msg  = "!!THERE IS SOMETHING WRONG WITH THE CURRENT "
            msg += "IMPLEMENTATION OF dFORCE LOSS / dPARAMETERS!!"
            raise UserWarning(msg)
            make_graph = True
    else:
        weightE = 1.
    if not type(p) is torch.Tensor:
        p = torch.as_tensor(p.tolist(), device=device)
    p = p.reshape((len(popt_list),-1))
    E, Eat, q, p, coordinates = run_custom_seqc(p, calculator=calculator,
                                coordinates=coordinates, species=species,
                                custom_params=popt_list)
    deltaE = Eat - Eref
    L = weightE * deltaE*deltaE
    if with_forces:
        g = agrad(E, coordinates, create_graph=make_graph)[0][0]
        F = torch.sum(g, dim=0) - g  # remove COM force
        deltaF2 = torch.square(F - Fref).sum()
        Lf = weightF * deltaF2
        L = L + Lf
    if jac:
        dLdp = agrad(L, p)[0]
        return dLdp.detach().flatten()
    return float(L)
    
def atomization_loss_jac(p, popt_list=[], calculator=None,coordinates=None,
              species=None, Eref=0., Fref=None, with_forces=False,
              weightE=1., weightF=0., jac=True):
    dLdp = atomization_loss(p, popt_list=popt_list, calculator=calculator,
              coordinates=coordinates, species=species, Eref=Eref,
              Fref=Fref, with_forces=with_forces, weightE=weightE,
              weightF=weightF, jac=True)
    return dLdp
