import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.seqm_functions.parameters import params
from seqm.basics import Energy, Parser
import seqm
seqm.seqm_functions.scf_loop.debug = False
seqm.seqm_functions.diag.DEGEN_EIGENSOLVER = True
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 400
seqm.seqm_functions.scf_loop.SCF_IMPLICIT_BACKWARD = True
seqm.seqm_functions.scf_loop.SCF_BACKWARD_ANDERSON_TOLERANCE = 1e-4
seqm.seqm_functions.scf_loop.SCF_BACKWARD_ANDERSON_MAXITER = 40
seqm.seqm_functions.scf_loop.SCF_BACKWARD_ANDERSON_HISTSIZE = 5

seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 200


torch.set_default_dtype(torch.double) # safe choice for finite differences in gradcheck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



prop = "energy"#"gap"
variable = "coords"#"param"
uhf = True

if uhf:
    species = torch.as_tensor([[6,1,1,1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor([
                  [
                   [1.22732374,          0.0000,              0.0000],
                   [0.0000,              0.0000,              0.0000],
                   [1.8194841064614802,  0.95341263319067747, 0.0000],
                   [1.7193342232738994, -0.93951967178254525, 0.0000]
                  ]
                 ], requires_grad=True, device=device)
else:
    species = torch.as_tensor([[8,6,1,1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor([
                  [
                   [0.0000,              0.0000,              0.0000],
                   [1.22732374,          0.0000,              0.0000],
                   [1.8194841064614802,  0.93941263319067747, 0.0000],
                   [1.8193342232738994, -0.93951967178254525, 0.0000]
                  ]
                 ], requires_grad=True, device=device)

prop2idx = {"energy":0, "gap":6}
elements = [0]+sorted(set(species.reshape(-1).tolist()))
if variable == "coords":
    lpar = []
elif variable == "param":
    lpar = ["zeta_s"]
else:
    raise ValueError("GradCheck not implemented for variable '"+variable+"'.")

seqm_parameters = {
                   'method'            : 'AM1',
                   'scf_eps'           : 1e-9,
                   'scf_converger'     : [0,0.25],
                   'sp2'               : [False, 1e-5],
                   'elements'          : elements,
                   'learned'           : lpar,
                   'pair_outer_cutoff' : 1e10, 
                   'scf_backward'      : 0,
                   'scf_backward_eps'  : 1e-9,
                   'parameter_file_dir': '/home/martin/work/software/PYSEQM/seqm/params/',
                   'eig'               : prop=="gap",
                   'UHF'               : uhf,
                  }

const = Constants().to(device)
S = 2 if uhf else 1
mol = Molecule(const, seqm_parameters, coordinates, species, mult=S, charges=0)
if variable == 'param':
    p = params(method=seqm_parameters['method'],
               elements=seqm_parameters['elements'],
               root_dir=seqm_parameters['parameter_file_dir'],
               parameters=seqm_parameters['learned'],).to(device)
    p, = p[mol.Z].transpose(0,1).contiguous()
    p.requires_grad_(True)
    learnedpar = {seqm_parameters['learned'][0]:p}
else:
    learnedpar = dict()


def parse_exc(msg):
    etext = str(msg)
    Jtext = etext.split("numerical:")[-1]
    [Jnum, Jana] = [eval("torch."+cmd) for cmd in Jtext.split("analytical:")]
    dJ = (Jnum - Jana).abs().max()
    rJ = torch.where(Jnum.abs()>1e-6, dJ / Jnum, 0.).abs().max()
    print("Delta max = ",dJ)
    print("rel D max = ",rJ)

gradvar = p if variable=="param" else coordinates
with torch.autograd.set_detect_anomaly(True):
    print("scf_backward = 0")
    eng = Energy(seqm_parameters).to(device)
    def f0(x): return eng(mol, learned_parameters=learnedpar, all_terms=True)[prop2idx[prop]]
    try:
        test_grad0 = torch.autograd.gradcheck(f0, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Gradient correct")
    except BaseException as eg0:
        print("Gradient NOT correct")
        parse_exc(eg0)
    try:
        test_hess0 = torch.autograd.gradgradcheck(f0, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Second derivative correct")
    except BaseException as eh0:
        print("Second derivative NOT correct")
        parse_exc(eh0)
    
    seqm_parameters['scf_backward'] = 1
    print("\nscf_backward = 1")
    eng = Energy(seqm_parameters).to(device)
    def f1(x): return eng(mol, learned_parameters=learnedpar, all_terms=True)[prop2idx[prop]]
    try:
        test_grad1 = torch.autograd.gradcheck(f1, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Gradient correct")
    except BaseException as eg1:
        print("Gradient NOT correct")
        parse_exc(eg1)
    try:
        test_hess1 = torch.autograd.gradgradcheck(f1, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Second derivative correct")
    except BaseException as eh1:
        print("Second derivative NOT correct")
        parse_exc(eh1)
    
    seqm_parameters['scf_backward'] = 2
    print("\nscf_backward = 2")
    eng = Energy(seqm_parameters).to(device)
    def f2(x): return eng(mol, learned_parameters=learnedpar, all_terms=True)[prop2idx[prop]]
    try:
        test_grad2 = torch.autograd.gradcheck(f2, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Gradient correct")
    except BaseException as eg2:
        print("Gradient NOT correct")
        parse_exc(eg2)
    try:
        test_hess2 = torch.autograd.gradgradcheck(f2, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Second derivative correct")
    except BaseException as eh2:
        print("Second derivative NOT correct")
        parse_exc(eh2)
    
