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
    E : torch.tensor, shape (1,)
        total SCF energy of system (differentiable instance)
    p : torch.tensor, shape (#custom parameters, nAtoms)
        custom parameters (differentiable instance)
    coordinates : torch.tensor, shape (nAtoms, 3)
        atomic positions (differentiable instance)
    
    """
    const = Constants().to(device)
    coordinates.requires_grad_(True)
    p.requires_grad_(True)
    learnedpar = {pname:p[i] for i, pname in enumerate(custom_params)}

    Eel, EnucAB, _, nconv = calculator(const, coordinates, species,
                                       learnedpar)
    Etot = Eel.sum() + EnucAB.sum()

    return Etot, p, coordinates
    

def seqc_loss(p, popt_list=[], calculator=None, coordinates=None,
              species=None, Eref=0., Fref=None, with_forces=False,
              weightE=1., weightF=0., jac=False):
    if with_forces:
        Fref = torch.as_tensor(Fref, device=device)
        sum_w = weightE + weightF
        weightE, weightF = weightE/sum_w, weightF/sum_w
    else:
        weightE = 1.
    if not type(p) is torch.Tensor:
        p = torch.as_tensor(p.tolist(), device=device)
    p = p.reshape((len(popt_list),-1))
    E, p, coordinates = run_custom_seqc(p, calculator=calculator,
                          coordinates=coordinates, species=species,
                          custom_params=popt_list)
    deltaE = E - Eref
    L = weightE * deltaE*deltaE
    if with_forces:
        if jac:
            msg  = "!!THERE IS SOMETHING WRONG WITH THE CURRENT "
            msg += "AUTOGRAD IMPLEMENTATION OF dLOSS/dPARAMETERS!!"
            raise UserWarning(msg)
            make_graph = True
        else:
            make_graph = False
        F = agrad(E, coordinates, create_graph=make_graph)[0][0]
        deltaF = torch.norm(F - Fref)
        L = L + weightF * deltaF
    if jac:
        dLdp = agrad(L, p)[0]
        return dLdp.flatten()
    return float(L)
    
def seqc_loss_jac(p, popt_list=[], calculator=None,coordinates=None,
              species=None, Eref=0., Fref=None, with_forces=False,
              weightE=1., weightF=0., jac=True):
    dLdp = seqc_loss(p, popt_list=popt_list, calculator=calculator,
              coordinates=coordinates, species=species, Eref=Eref,
              Fref=Fref, with_forces=with_forces, weightE=weightE,
              weightF=weightF, jac=True)
    return dLdp
