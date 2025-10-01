import torch
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.seqm_functions.parameters import params
from seqm.basics import Energy, Parser
from seqm.seqm_functions.occupations import fractional_occ

torch.set_default_dtype(torch.double) # safe choice for finite differences in gradcheck
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



variable = "eps"
uhf = False
kT = torch.tensor(1.5, requires_grad=True)
smearing = "fermi"

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
    n_el = torch.tensor([[4,3]])
else:
    species = torch.as_tensor([[8,6,1,1]], dtype=torch.int64, device=device)
    coordinates = torch.tensor([
                  [
                   [0.0000,              0.0000,              0.0000],
                   [1.22832374,          0.0000,              0.0000],
                   [1.8194841064614802,  0.93941263319067747, 0.0000],
                   [1.8193342232738994, -0.93951967178254525, 0.0000]
                  ]
                 ], requires_grad=True, device=device)
    n_el = torch.tensor([6])

elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method'            : 'AM1',
                   'scf_eps'           : 1e-9,
                   'scf_converger'     : [4, {}],
                   'sp2'               : [False, 1e-5],
                   'elements'          : elements,
                   'learned'           : [],
                   'pair_outer_cutoff' : 1e10, 
                   'scf_backward'      : 0,
                   'scf_backward_eps'  : 1e-9,
                   'parameter_file_dir': '/home/martin/work/software/PYSEQM/seqm/params/',
                   'eig'               : True,
                   'UHF'               : uhf,
                   'occ_mode'          : 2,
                   'occ_kT'            : kT,
                   'smearing'          : smearing,
                  }

const = Constants().to(device)
S = 2 if uhf else 1
mol = Molecule(const, seqm_parameters, coordinates, species, mult=S, charges=0)

def parse_exc(msg):
    etext = str(msg)
    Jtext = etext.split("numerical:")[-1]
    [Jnum, Jana] = [eval("torch."+cmd) for cmd in Jtext.split("analytical:")]
    dJ = (Jnum - Jana).abs().max()
    rJ = torch.where(Jnum.abs()>1e-6, dJ / Jnum, 0.).abs().max()
    print("Delta max = ",dJ)
    print("rel D max = ",rJ)

with torch.autograd.set_detect_anomaly(True):
    seqm_parameters['scf_backward'] = 1
    print("\nscf_backward = 1")
    eng = Energy(seqm_parameters).to(device)
    res = eng(mol, learned_parameters={}, all_terms=True)
    eps = res[7]
    print("Occupations", fractional_occ(eps, res[8][0].shape, n_el, kT=kT, smearing=smearing).round(decimals=2).tolist())
    eps_in = eps.detach().clone()
    eps_in.requires_grad_(True)
    def f1(x): return fractional_occ(x, res[8][0].shape, n_el, kT=kT, smearing=smearing)
    try:
        test_grad1 = torch.autograd.gradcheck(f1, (eps_in,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Gradient correct")
    except BaseException as eg1:
        print("Gradient NOT correct")
        parse_exc(eg1)
    try:
        test_hess1 = torch.autograd.gradgradcheck(f1, (eps_in,), eps=1e-6, atol=0.001, rtol=0.01)
        print("Second derivative correct")
    except BaseException as eh1:
        print("Second derivative NOT correct")
        parse_exc(eh1)
    
#    seqm_parameters['scf_backward'] = 2
#    print("\nscf_backward = 2")
#    eng = Energy(seqm_parameters).to(device)
#    def f2(x): return eng(mol, learned_parameters=learnedpar, all_terms=True)[prop2idx[prop]]
#    try:
#        test_grad2 = torch.autograd.gradcheck(f2, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
#        print("Gradient correct")
#    except BaseException as eg2:
#        print("Gradient NOT correct")
#        parse_exc(eg2)
#    try:
#        test_hess2 = torch.autograd.gradgradcheck(f2, (gradvar,), eps=1e-6, atol=0.001, rtol=0.01)
#        print("Second derivative correct")
#    except BaseException as eh2:
#        print("Second derivative NOT correct")
#        parse_exc(eh2)
    
