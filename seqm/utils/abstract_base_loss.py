#############################################################################
# Base class and functionality of loss modules for SEQC                     #
#  - AbstractLoss: abstract base class as template for loss modules         #
#                                                                           #
# Current (Feb/07)                                                          #
# TODO: . typing                                                            #
#       . learning rate scheduler!                                          #
#       . enable loss functions from pytroch (at the moment: custom RSS)    #
#       . check functionality of minimize with other (pytorch) optimizers   #
#       . double-check GPU support!                                         #
#############################################################################

import torch
from warnings import warn
from abc import ABC, abstractmethod
from torch.autograd import grad as agrad
from .pyseqm_helpers import prepare_array, Orderator


torch.set_default_dtype(torch.float64)
prop2index = {'atomization':0, 'energy':1, 'forces':2, 'gap':3}


class RSSperAtom(torch.nn.Module):
    def __init__(self):
        super(RSSperAtom, self).__init__()
        self.requires_grad = False
    
    def forward(self, prediction, reference, nAtoms=1, intensive=False):
        delta2 = torch.square(prediction - reference)
        if intensive: return delta2.sum()
        sumaxes = tuple(range(prediction.dim()))[1:]
        return (delta2.sum(dim=sumaxes) / nAtoms).sum()
        
    

class AbstractLoss(ABC, torch.nn.Module):
    """
    Abstract base class for loss modules.
    Implements common features such as basic initialization, loss evaluation,
    adding loss properties, and minimization.
    
    Individual, concrete loss modules have to extend base `__init__`
    and provide a corresponding implementation of `self.run_calculation(x)`.
    """
    def __init__(self, species, coordinates, custom_params=None, 
                 loss_type="RSSperAtom", loss_args=(), loss_kwargs={}):
        super(AbstractLoss, self).__init__()
        ## initialize parent module and attributes
        self.implemented_properties = ['atomization','energy','forces','gap']
        self.n_implemented = len(self.implemented_properties)
        self.weights = torch.zeros(self.n_implemented)
        self.is_intensive = [False, False, False, True]
        self.include = []
        self.implemented_loss_types = ["RSSperAtom"]
        
        ## collect attributes from input
        if loss_type not in self.implemented_loss_types:
            raise ValueError("Unknown loss type '"+loss_type+"'.")
        exec("self.loss_func = "+loss_type+"(*loss_args, **loss_kwargs)")
        self.orderer = Orderator(species, coordinates)
        self.species, self.coordinates = self.orderer.prepare_input()
        if not (self.species == prepare_array(species, "Z")).all():
            msg  = "For efficiency reasons and consistency, all individual "
            msg += "molecules (along with their positions and reference "
            msg += "forces) have to be ordered according to descending "
            msg += "atomic numbers. Hint: seqm.utils.pyseqm_helpers.Orderator"
            raise ValueError(msg)
        self.nMols = self.species.shape[0]
        self.nAtoms = torch.count_nonzero(self.species, dim=1)
        self.req_shapes = {'forces':self.coordinates.shape}
        for prop in ['energy', 'atomization', 'gap']:
            self.req_shapes[prop] = (self.nMols,)
        self.custom_params = custom_params
        
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
        
    def forward(self, x):
        """ Get Loss. """
        Deltas = torch.zeros(self.n_implemented)
        res = self.run_calculation(x)
        for i in self.include:
            ref = eval("self."+self.implemented_properties[i]+"_ref")
            Deltas[i] = self.loss_func(res[i], ref, nAtoms=self.nAtoms, 
                                       intensive=self.is_intensive[i])
        return (Deltas * self.weights).sum()
    
    @abstractmethod
    def run_calculation(self, x):
        """
        Abstract method for gathering results in the form Eat, Etot, F, gap.
        This has to be implemented by the individual concrete loss modules
        (either as explicit function or during `__init__` see, e.g., below).
        `run_calculation` should return inf for molecules where SCF failed!
        """
        pass
    
    def add_loss(self, prop, prop_ref, weight=1.):
        """
        Add individual loss evaluators as defined above to loss function.
        If implementing a new property, please add loss functon
        `<property>_loss(...)` above and update self.implemented_properties
        """
        if prop not in self.implemented_properties:
            msg  = "Only '"+"', '".join(self.implemented_properties)
            msg += "' implemented for loss. Check for typos or write "
            msg += "corresponding loss function for '"+prop+"'."
            raise ValueError(msg)
        if prop == 'gap':
            msg  = 'HOMO-LUMO gap explicitly depends on eigenvalues. '
            msg += 'These might have derivative discontinuities w.r.t. '
            msg += 'SEQM parameters (MOs crossing) -> unlikely, but '
            msg += 'possible instabilities in autograd!'
            warn(msg)
        self.weights[prop2index[prop]] = weight
        if prop == 'forces':
            ref_proc = prepare_array(prop_ref, prop+'_reference')
        elif torch.is_tensor(prop_ref):
            ref_proc = prop_ref
        elif type(prop_ref) in [int, float]:
            ref_proc = torch.tensor([prop_ref])
        elif type(prop_ref) == list:
            ref_proc = torch.tensor(prop_ref)
        msg  = "Reference "+prop+" of shape "+str(tuple(ref_proc.shape))
        msg += " doesn't match input structure of "+str(self.nMols)
        msg += " molecule(s)!"
        assert ref_proc.shape == self.req_shapes[prop], msg
        exec('self.'+prop+'_ref = ref_proc')
        self.include.append(prop2index[prop])
    
    def minimize(self, x, n_epochs=4, optimizer='LBFGS', **kwargs):
        """ Generic minimization routine. """
        try:
            my_opt = getattr(torch.optim, optimizer)
        except AttributeError:
            msg  = "Unknown optimizer '"+optimizer+"'. Currently, only "
            msg += "optimizers from torch.optim are supported."
            raise ImportError(msg)
        opt_options = {'max_iter':10, 'tolerance_grad':1e-06,
                       'tolerance_change':1e-08}
        opt_options.update(kwargs)
        opt = my_opt([x], **opt_options)
        def closure():
            if torch.is_grad_enabled(): opt.zero_grad()
            L = self(x)
            L.backward()
            return L
        self.minimize_log = []
        for n in range(n_epochs):
            L = opt.step(closure)
            self.minimize_log.append(L.item())
        return x, L
    
    def gradient(self, x):
        """ Return gradient of loss at input x. """
        L = self(x)
        dLdx = agrad(L, x, retain_graph=True)[0]
        return dLdx
        
    

