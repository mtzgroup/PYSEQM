import torch
from seqm.seqm_functions.constants import Constants
from seqm.basics import  Energy
import seqm
seqm.seqm_functions.scf_loop.debug = False
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 50
torch.set_default_dtype(torch.double)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device)
coordinates = torch.tensor([
                  [
              [0.0000,              0.0000,              0.0000],
              [1.22732374,          0.0000,              0.0000],
              [1.8194841064614802,  0.93941263319067747, 0.0000],
              [1.8193342232738994, -0.93951967178254525, 0.0000]
                  ]
                 ], device=device)
elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
           'method' : 'AM1',  # AM1, MNDO, PM#
           'scf_eps' : 1.0e-6,  # unit eV, change of electric energy,
           'scf_converger' : [1,0.25], # converger used for scf loop
           'sp2' : [False, 1.0e-5],  # whether to use sp2 in scf loop,
           'elements' : elements, #[0,1,6,8],
           'learned' : [], # learned parameters names, e.g ['U_ss']
           'pair_outer_cutoff' : 1.0e10, # consistent with the unit 
           'scf_backward': 1,
           'scf_backward_eps': 1.0e-6,
                   }

const = Constants().to(device)
with torch.autograd.set_detect_anomaly(True):
    eng = Energy(seqm_parameters).to(device)
    x0 = coordinates.reshape(-1)
    x0.requires_grad_(True)
    def func(x):
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, \
        notconverged = eng(const, x.reshape(1,-1,3), species, 
                           all_terms=True)
        return Hf
    func(x0)
    #grad = torch.autograd.grad(Etot.sum(), coordinates, create_graph=True, retain_graph=True)[0]
    hess1 = torch.autograd.functional.hessian(func, x0)
    print("Backward=1:  absmax(H - H.T)     = ",(hess1-hess1.transpose(0,1)).abs().max().item())
    
    torch.autograd.gradcheck(eng.apply, coordinates)
    

seqm_parameters['scf_backward'] = 2
with torch.autograd.set_detect_anomaly(True):
    eng = Energy(seqm_parameters).to(device)
    x0 = coordinates.reshape(-1)
    x0.requires_grad_(True)
    def func(x):
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, \
        notconverged = eng(const, x.reshape(1,-1,3), species,
                           all_terms=True)
        return Hf
    func(x0)
    hess2 = torch.autograd.functional.hessian(func, x0)
    dH = (hess1-hess2).abs()
    print("Backward=2:  absmax(H - H.T)     = ",(hess2-hess2.transpose(0,1)).abs().max().item())
    print("absmax(backward=1 - backward=2)  = ",dH.max().item())
    print("absmean(backward=1 - backward=2) = ",dH.mean().item())

    
