#############################################################################
# Base class and functionality of loss modules for SEQC                     #
#  - NoScheduler: dummy scheduler (simple pass)                             #
#  - RSSperAtom: residual sum of squares loss (weighted by number of atoms  #
#                for extensive properties                                   #
#  - AbstractLoss: abstract base class as template for loss modules         #
#                                                                           #
# Current (Feb/08)                                                          #
# TODO: . typing                                                            #
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


class NoScheduler:
    """ Dummy scheduler class. """
    def __init__(self): pass
    def step(self): pass

    
class RSSperAtom(torch.nn.Module):
    """
    Basic implementation of residual sum of squares loss.
    
    Parameters:
    -----------
      . prediction, torch.Tensor: predicted property for each molecule
      . reference, torch.Tensor: corresponding reference value for property
      . intensive, bool: whether property is intensive or not
    
    Returns:
    --------
      . residual sum of squares
          if intensive: sum_{i,...} (predicted_{i,...} - reference_{i,...})^2
          else: sum_A (sum_{i,...} (predicted_{A,i,...} 
                                      - reference_{A,i,...})^2) / nAtoms_A
    """
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
        ## initialize parent modules and attributes
        super(AbstractLoss, self).__init__()
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
        # check for correctly ordered input (avoid re-ordering at runtime)
        if not (self.species == prepare_array(species, "Z")).all():
            msg  = "For efficiency reasons and consistency, all individual "
            msg += "molecules (along with their positions and reference "
            msg += "forces) have to be ordered according to descending "
            msg += "atomic numbers. Hint: seqm.utils.pyseqm_helpers.Orderator"
            raise ValueError(msg)
        # set required shapes for reference data
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
        """
        Loss evaluation. Calls calculation according to concrete loss model
        and combines losses for individual properties.
        """
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
        `run_calculation` should return inf for molecules where SCF failed!
        """
        pass
        
    
    def add_loss(self, prop, prop_ref, weight=1.):
        """
        Add individual properties to include in total loss function.
        If implementing a new property, update self.implemented_properties, 
        prop2index, and Deltas in `self.forward`, as well as `run_calculation`
        in concrete modules!
        
        Parameters:
        -----------
          . prop, str: name of property (as in self.implemented_properties)
          . prop_ref, torch.Tensor: reference values for property
          . weight, float: weighting factor for property in total loss
        
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
        
    
    def minimize(self, x, n_epochs=4, optimizer="LBFGS", upward_thresh=5,
                 opt_kwargs={}, scheduler=None, scheduler_kwargs={}):
        """
        Generic routine for minimizing loss.
        
        Parameters:
        -----------
          . x, torch.Tensor: initial guess of parameters entering self.forward
          . n_epochs, int: number of epochs in optimization
          . optimizer, str: optimizer from torh.optim for running minimization
                default: LBFGS
          . upward_thresh, int: number of consecutive increasing loss to 
                accept during minimization, default: 5
          . opt_kwargs, dict: dictionary of kwargs for optimizer, default: {}
          . scheduler, str/list of str: learning rate scheduler(s) from 
                torch.optim.lr_scheduler, default: None
          . scheduler_kwargs, {}/list of {}: kwargs for scheduler(s)
        
        Returns:
        --------
          . x, torch.tensor: result of optimization
          . L, torch.Tensor: optimal loss corresponding to x
        
        Sets:
        -----
          . self.minimize_log, list: history of loss at every epoch
        
        """
        try:
            my_opt = getattr(torch.optim, optimizer)
        except AttributeError:
            msg  = "Unknown optimizer '"+optimizer+"'. Currently, only "
            msg += "optimizers from torch.optim are supported."
            raise ImportError(msg)
        opt_options = {'max_iter':10, 'tolerance_grad':1e-06,
                       'tolerance_change':1e-08, 'lr':0.5}
        opt_options.update(opt_kwargs)
        opt = my_opt([x], **opt_options)
        lr_sched = self.add_scheduler(scheduler, opt, scheduler_kwargs)
        def closure():
            if torch.is_grad_enabled(): opt.zero_grad()
            L = self(x)
            L.backward()
            return L
        Lbak, n_up, self.minimize_log = torch.inf, 0, []
        for n in range(n_epochs):
            L = opt.step(closure)
            n_up = (L > Lbak).float() * (n_up + 1)
            if n_up > upward_thresh:
                msg  = "Loss increased more than "+str(upward_thresh)
                msg += " times. This may indicate a failure in training. "
                msg += "You can adjust this limit via 'upward_thresh'."
                raise RuntimeError(msg)
            Lbak = L.clone()
            for sched in lr_sched: sched.step()
            self.minimize_log.append(L.item())
        return x, L
        
    
    def add_scheduler(self, scheduler, optimizer, sched_kwargs={}):
        """
        Adds learning rate scheduler(s) to optimizer in self.minimize.
        
        Parameters:
        -----------
          . scheduler, str/list of str: name(s) of learning rate scheduler(s)
              as contained in torch.optim.lr_scheduler
          . optimizer, torch.optim.Optimizer: optimizer to attach scheduler to
          . sched_kwargs, {}/ list of {}: kwargs for scheduler
        
        Returns:
        --------
          . lr_sched, list of torch.optim.lr_scheduler attached to optimizer
        
        """
        if scheduler is None:
            return [NoScheduler()]
        elif isinstance(scheduler, list) and len(scheduler)>1:
            lr_sched = []
            for i, s in enumerate(scheduler):
                try:
                    my_sched = getattr(torch.optim.lr_scheduler, s)
                    lr_sched.append(my_sched(optimizer, **sched_kwargs[i]))
                except AttributeError:
                    msg  = "Unknown scheduler '"+sched+"'. Currently, only "
                    msg += "schedulers in torch.optim.lr_scheduler supported"
                    raise ImportError(msg)
            return lr_sched
        elif isinstance(scheduler, str):
            try:
                my_sched = getattr(torch.optim.lr_scheduler, scheduler)
                return [my_sched(optimizer, **sched_kwargs)]
            except AttributeError:
                msg  = "Unknown scheduler '"+scheduler+"'. Currently, only "
                msg += "schedulers in torch.optim.lr_scheduler supported"
                raise ImportError(msg)
        else:
            msg = "Unrecognized type '"+type(scheduler)+"' for 'scheduler'"
            raise ValueError(msg)
        
    
    def gradient(self, x):
        """ Return gradient of loss w.r.t. input x at x. """
        L = self(x)
        dLdx = agrad(L, x, retain_graph=True)[0]
        return dLdx
        
    

