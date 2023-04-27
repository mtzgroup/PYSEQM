#get force y component
#move atom 0 along y direction, get a function fy(y)
#get dfy/dy

import torch
import numpy as np

from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.basics import Energy
import seqm
seqm.seqm_functions.scf_loop.debug = True
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 200
seqm.seqm_functions.scf_loop.RAISE_ERROR_IF_SCF_BACKWARD_FAILS = False
seqm.seqm_functions.scf_loop.SCF_IMPLICIT_BACKWARD = False
seqm.seqm_functions.MAX_ITER_TO_STOP_IF_SCF_BACKWARD_DIVERGE = 20
seqm.seqm_functions.DEGEN_EIGENSOLVER = False


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


idir = 0

for backmode in [0,1,2]:
    N = 200

    #"""
    #pertubation direction
    dir1 = torch.tensor([0.,0.,0.])#torch.randn(3).to(device)
    dir1[idir] = 1.
    dir1 /= torch.norm(dir1)
    
    #force direction
    dir2 = torch.tensor([0.,0.,0.])#torch.randn(3).to(device)
    dir2[idir] = 1.
    dir2 /= torch.norm(dir2)
    
    dxmin = -0.5
    dxmax = 0.5
    
    N += 1
    dx = torch.arange(N+0.0,device=device)*(dxmax-dxmin)/(N-1.0)+dxmin
    
    const = Constants().to(device)
    
    species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device) \
                   .expand(N,4)
    
    coordinates_op = np.array([
                 [
                  [0.014497983896917479, 3.208059775069048e-05, -1.0697192017402962e-07],
                  [1.3364260303072648, -3.2628339194439124e-05, 8.51016890853131e-07],
                  [1.757659914731728, 1.03950803854101, -5.348699815983099e-07],
                  [1.7575581407994696, -1.039614529391432, 2.84735846426227e-06]
                 ],
                 ])
    coordinates = torch.tensor(coordinates_op.repeat(N, axis=0), device=device)
    coordinates[...,0,:] += dx.unsqueeze(1)*dir1.unsqueeze(0)
    
    
    elements = [0]+sorted(set(species.reshape(-1).tolist()))
    seqm_parameters = {
                       'method' : 'AM1',  # AM1, MNDO, PM#
                       'scf_eps' : 1e-9,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                       'scf_converger' : [0,0.2], # converger used for scf loop
                                             # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                             # [1], adaptive mixing
                                             # [2], adaptive mixing, then pulay
                       'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                                #[True, eps] or [False], eps for SP2 conve criteria
                       'elements' : elements, #[0,1,6,8],
                       'learned' : [], # learned parameters name list, e.g ['U_ss']
                       'parameter_file_dir' : '../../seqm/params/', # file directory for other required parameters
                       'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                       'eig' : True,
                       'scf_backward': backmode, #scf_backward==0: ignore the gradient on density matrix
                                          #scf_backward==1: use recursive formu
                                          #scf_backward==2: go backward scf loop directly
                       'scf_backward_eps' : 1e-9,
                       }
    
    coordinates.requires_grad_(True)
    mol = Molecule(const, seqm_parameters, coordinates, species)
    
    with torch.autograd.set_detect_anomaly(True):
        eng = Energy(seqm_parameters).to(device)
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, gap, e, P, charge, notconverged = eng(mol, all_terms=True)
        force, = torch.autograd.grad(Etot.sum(), coordinates, create_graph=True)
        
        Fdir2 = -force[:,0,idir]
        dFdd, = torch.autograd.grad(Fdir2.sum(), coordinates)
        dFdir2_ddir1 = dFdd[:,0,idir]
        
    
    f = open('log'+str(backmode)+'.dat', 'w')
    f.write("#index, dx (Angstrom), Fdir2 (eV/Angstrom), dFdir2_ddir1, Etot\n")
    for i in range(N):
        f.write("%d %12.8e %12.8e %12.8e %12.8e\n" % (i,dx[i],Fdir2[i], dFdir2_ddir1[i], Etot[i] ))
    f.close()
