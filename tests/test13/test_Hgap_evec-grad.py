import torch
import numpy as np

from copy import deepcopy as dcopy
from seqm.seqm_functions.constants import Constants
from seqm.seqm_functions.parameters import params
from seqm.basics import Parser, Pack_Parameters, Energy
import seqm
#seqm.seqm_functions.scf_loop.debug = True
seqm.seqm_functions.scf_loop.SCF_BACKWARD_MAX_ITER = 15
#seqm.seqm_functions.diag.DEGEN_EIGENSOLVER = False # (default: True)

from os import path

here = path.abspath(path.dirname(__file__))

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

const = Constants().to(device)

degen = True
changeAt = 2
nAt = 5 if degen else 4

if degen:
    species = torch.as_tensor([[6,1,1,1,1]],dtype=torch.int64, device=device)
    coordinates = torch.tensor([
             [
                    [ 0.000000,  0.000000,  0.000000],
                    [-0.629118,  0.629118, -0.629118],
                    [ 0.629118, -0.629118, -0.629118],
                    [-0.629118, -0.629118,  0.629118],
                    [ 0.629118,  0.629118,  0.629118],
             ],
             ], device=device)
else:
    species = torch.as_tensor([[8,6,1,1]],dtype=torch.int64, device=device)
    coordinates = torch.tensor([
             [
              [0.014497983896917479, 3.208059775069048e-05, -1.0697192017402962e-07],
              [1.3364260303072648, -3.2628339194439124e-05, 8.51016890853131e-07],
              [1.757659914731728, 1.03950803854101, -5.348699815983099e-07],
              [1.7575581407994696, -1.039614529391432, 2.84735846426227e-06]
             ],
             ], device=device)


elements = [0]+sorted(set(species.reshape(-1).tolist()))
seqm_parameters = {
                   'method' : 'AM1',  # AM1, MNDO, PM#
                   'scf_eps' : 1.0e-6,  # unit eV, change of electric energy, as nuclear energy doesnt' change during SCF
                   'scf_converger' : [0,0.15], # converger used for scf loop
                                         # [0, 0.1], [0, alpha] constant mixing, P = alpha*P + (1.0-alpha)*Pnew
                                         # [1], adaptive mixing
                                         # [2], adaptive mixing, then pulay
                   'sp2' : [False, 1.0e-5],  # whether to use sp2 algorithm in scf loop,
                                            #[True, eps] or [False], eps for SP2 conve criteria
                   'elements' : elements, #[0,1,6,8],
                   'learned' : ['U_ss'], # learned parameters name list, e.g ['U_ss']
                   'parameter_file_dir' : here+'/../../seqm/params/', # file directory for other required parameters
                   'pair_outer_cutoff' : 1.0e10, # consistent with the unit on coordinates
                   'eig' : True,
                   'scf_backward': 2, #scf_backward==0: ignore the gradient on density matrix
                                      #scf_backward==1: use recursive formula
                                      #scf_backward==2: go backward scf loop directly
                   'scf_backward_eps' : 1.0e-3,
                   }

#parser is not needed here, just use it to get Z and create "learned" parameters
#prepare a fake learned parameters: learnedpar
parser = Parser(seqm_parameters).to(device)
nmol, molsize, \
nHeavy, nHydro, nocc, \
Z, maskd, atom_molid, \
mask, pair_molid, ni, nj, idxi, idxj, xij, rij = parser(const, species, coordinates)
#add learned parameters
#here use the data from mopac as example
homo, lumo = nocc[0]-1, nocc[0]

p = params(method=seqm_parameters['method'],
           elements=seqm_parameters['elements'],
           root_dir=seqm_parameters['parameter_file_dir'],
           parameters=seqm_parameters['learned'],).to(device)
p, = p[Z].transpose(0,1).contiguous()
p.requires_grad_(True)
learnedpar = {seqm_parameters['learned'][0]:p}
    
with torch.autograd.set_detect_anomaly(True):
    def func(x):
        eng = Energy(seqm_parameters).to(device)
        Hf, Etot, Eelec, Enuc, Eiso, EnucAB, e, P, charge, notconverged = eng(const, coordinates, species, learned_parameters=learnedpar, all_terms=True)
        G = e[:,lumo]-e[:,homo]
        dG = (G - 11.5)**2
        L = dG.sum()
        return L

    func(p)
    H = torch.autograd.functional.hessian(func, p)
    e, v = torch.linalg.eigh(H, UPLO='U')

print(e)
print("")
print(v)

raise SystemExit

#tg = sum(p.grad[changeAt::nAt] for changeAt in [1,2,3,4])
#tg = p.grad[changeAt::nAt]

f = open('log.dat', 'w')
f.write("#index, par, gap [eV], grad\n")
for i in range(N):
    f.write( "%d %12.8e %12.8e %12.8e\n" % (i,t[i],dG[i],tg[i]) )
f.close()
        
#        grads.append(tg[i0])
#    print(thresh,': ',grads)

eorb = e.detach().numpy()
eorb = np.where((eorb==0), np.nan, eorb)
t = t.detach().numpy()[::-1]
eorb = eorb[::-1]
import matplotlib.pyplot as plt
ax = plt.subplot2grid((1,1),(0,0))
ax.plot(t, eorb, c='k', lw=0.85)
ax.plot(t, eorb[:,homo], c='tab:blue', label='HOMO')
ax.plot(t, eorb[:,lumo], c='tab:red',  label='LUMO')
ylim = ax.get_ylim()
ax.plot(t, eorb[:,0], c='k', lw=0.85, label=r'$\varepsilon_i$')
ax.plot([p0,p0], ylim, c='0.5',ls=':', lw=1, label=r'$p_{0}$')
ax.set_ylim(-32,ylim[1])
ax.set_xlim(t[0],t[-1])
ax.set_ylabel(r'$\varepsilon_{\mathsf{orb}}\,\,\mathsf{[eV]}$')
ax.set_xlabel(r'$p\,\,\mathsf{[arb.u.]}$')
ax.legend(loc='upper left', prop={'size':11})

fig = plt.gcf()
fig.set_size_inches(4,3)
fig.subplots_adjust(left=0.165, top=0.99, right=0.99, bottom=0.15)
fig.savefig('CH4_MOs-vs-p_LinComb.pdf')
plt.close()


