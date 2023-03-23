import torch
import torchviz
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
                 ], requires_grad=True, device=device)
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
    def func(x):
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, \
        notconverged = eng(const, coordinates, species, all_terms=True)
        return Hf
    test_grad1 = torch.autograd.gradcheck(func, coordinates, raise_exception=False)
    lab_grad1 = "correct" if test_grad1 else "incorrect"
    E = func(coordinates)
    grad, = torch.autograd.grad(E, coordinates, create_graph=True)
    hess, = torch.autograd.grad(grad.sum(), coordinates, create_graph=True)
    graph = torchviz.make_dot((hess, coordinates, E), show_saved=True)
    graph.format = 'svg'
    graph.render()
    test_hess1 = torch.autograd.gradgradcheck(func, coordinates, raise_exception=False)
    lab_hess1 = "correct" if test_hess1 else "incorrect"
    
    seqm_parameters['scf_backward'] = 2
    eng = Energy(seqm_parameters).to(device)
    def func(x):
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, \
        notconverged = eng(const, coordinates, species, all_terms=True)
        return Hf
    test_grad2 = torch.autograd.gradcheck(func, coordinates, raise_exception=False)
    lab_grad2 = "correct" if test_grad2 else "incorrect"
    test_hess2 = torch.autograd.gradgradcheck(func, coordinates, raise_exception=False)
    lab_hess2 = "correct" if test_hess2 else "incorrect"
    
    print("scf_backward=1")
    print("Gradients "+lab_grad1)
    print("Second derivatives "+lab_hess1)
    print("\nscf_backward=2")
    print("Gradients "+lab_grad2)
    print("Second derivatives "+lab_hess2)
    
