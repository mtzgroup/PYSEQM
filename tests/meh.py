import torch
from seqm.seqm_functions.constants import Constants
from seqm.basics import Force
import seqm
seqm.seqm_functions.scf_loop.debug = False
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 50
torch.set_default_dtype(torch.double) # safe choice for finite differences in gradcheck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


species = torch.as_tensor([[8,6,1,1]], dtype=torch.int64, device=device)
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
                   'method'            : 'AM1',
                   'scf_eps'           : 1.0e-6,
                   'scf_converger'     : [1,0.25],
                   'sp2'               : [False, 1.0e-5],
                   'elements'          : elements,
                   'learned'           : [],
                   'pair_outer_cutoff' : 1.0e10, 
                   'scf_backward'      : 1,
                   'scf_backward_eps'  : 1.0e-6,
                   '2nd_grad'          : True,
                   }

const = Constants().to(device)
with torch.autograd.set_detect_anomaly(True):
    calc = Force(seqm_parameters).to(device)
    def f1(x): return calc(const, coordinates, species, mode="new")[0].sum()
    test_grad1 = torch.autograd.gradcheck(f1, coordinates, raise_exception=False)
    lab_grad1 = "correct" if test_grad1 else "incorrect"
#    test_hess1 = torch.autograd.gradgradcheck(f1, coordinates, raise_exception=False)
#    lab_hess1 = "correct" if test_hess1 else "incorrect"
    
    seqm_parameters['scf_backward'] = 2
    calc = Force(seqm_parameters).to(device)
    def f2(x): return calc(const, coordinates, species, mode="new")[0].sum()
    test_grad2 = torch.autograd.gradcheck(f2, coordinates, raise_exception=False)
    lab_grad2 = "correct" if test_grad2 else "incorrect"
#    test_hess2 = torch.autograd.gradgradcheck(f2, coordinates, raise_exception=False)
#    lab_hess2 = "correct" if test_hess2 else "incorrect"
    
    print("scf_backward=1")
    print("Gradients "+lab_grad1)
#    print("Second derivatives "+lab_hess1)
    print("\nscf_backward=2")
    print("Gradients "+lab_grad2)
#    print("Second derivatives "+lab_hess2)
    
