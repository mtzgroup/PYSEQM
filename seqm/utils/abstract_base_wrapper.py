#############################################################################
# Base class and functionality of wrappers for SEQC calculations            #
#  - NoScheduler: dummy scheduler (simple pass)                             #
#  - AbstractWrapper: abstract base class as template for wrapper modules   #
#                                                                           #
# Current (Feb/12)                                                          #
# TODO: . typing                                                            #
#       . double-check GPU support!                                         #
#       . enable optimization of only select atoms/elements (mask grad)     #
#############################################################################

import torch
from warnings import warn
from abc import ABC, abstractmethod
from torch.autograd import grad as agrad
from .pyseqm_helpers import prepare_array, Orderator
from .loss_functions import *

torch.set_default_dtype(torch.float64)
# THIS NEEDS TO BE CONSISTENT WITH DATALOADERS!
prop2index = {'atomization':0, 'energy':1, 'forces':2, 'gap':3}


class NoScheduler:
    """ Dummy scheduler class. """
    def __init__(self): pass
    def step(self): pass


class AbstractWrapper(ABC, torch.nn.Module):
    """
    Abstract base class for loss modules.
    Implements common features such as basic initialization, loss evaluation,
    adding loss properties, and minimization.
    
    Individual, concrete loss modules have to extend base `__init__`
    and provide a corresponding implementation of `self.run_calculation(x)`.
    """
    def __init__(self, custom_params=None, loss_type="RSSperAtom",
                 loss_args=(), loss_kwargs={}):
        ## initialize parent modules and attributes
        super(AbstractWrapper, self).__init__()
        self.implemented_properties = ['atomization','energy','forces','gap']
        self.n_implemented = len(self.implemented_properties)
        self.is_extensive = [True, True, True, False]
        self.implemented_loss_types = ["RSSperAtom"]
        ## collect attributes from input
        if loss_type not in self.implemented_loss_types:
            raise ValueError("Unknown loss type '"+loss_type+"'.")
        exec("self.loss_func = "+loss_type+"(*loss_args, **loss_kwargs)")
        self.custom_params = custom_params
        
    
    def __eq__(self, other):
        if self.__class__ != other.__class__: return False
        return self.__dict__ == other.__dict__
        
    @abstractmethod
    def forward(self, x, species, coordinates):
        """
        Abstract method for running calculation and gathering results.
        This has to be implemented by the individual concrete wrappers
        """
        pass
        
    
    def train(self, x, dataloader, n_epochs=4, include=[], optimizer="LBFGS",
              opt_kwargs={}, upward_thresh=5, scheduler=None, 
              scheduler_kwargs={}, validation_loader=None):
        """
        Generic routine for minimizing loss.
        
        Parameters:
        -----------
          . x, torch.Tensor: initial guess of parameters entering self.forward
          . dataloader, torch.utils.data.Dataloader: dataloader for training
                (see seqm.utils.dataloaders)
          . n_epochs, int: number of epochs in optimization
          . include, list of str: list of properties to include in loss
          . optimizer, str: optimizer from torh.optim for running minimization
                default: LBFGS
          . opt_kwargs, dict: dictionary of kwargs for optimizer, default: {}
          . upward_thresh, int: number of consecutive increasing loss to 
                accept during minimization, default: 5
          . scheduler, str/list of str: learning rate scheduler(s) from 
                torch.optim.lr_scheduler, default: None
          . scheduler_kwargs, {}/list of {}: kwargs for scheduler(s)
          . validation_loader, torch.utils.data.Dataloader: dataloader for validation
                (see seqm.utils.dataloaders)
        
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
        self.opt = my_opt([x], **opt_kwargs)
        lr_sched = self.add_scheduler(scheduler, self.opt, scheduler_kwargs)
        if any(prop not in self.implemented_properties for prop in include):
            raise ValueError("Requested properties not available.")
        self.include_loss = sorted([prop2index[prop] for prop in include])
        n_train = len(dataloader)
        Lbak, n_up, self.minimize_log = torch.inf, 0, []
        for epoch in range(n_epochs):
            L_epoch = self.train_epoch(x, dataloader)
            L_avg = L_epoch / n_train
            n_up = int(L_epoch > Lbak) * (n_up + 1)
            if n_up > upward_thresh:
                msg  = "Loss increased more than "+str(upward_thresh)
                msg += " times. This may indicate a failure in training. "
                msg += "You can adjust this limit via 'upward_thresh'."
                raise RuntimeError(msg)
            Lbak = L_epoch
            if validation_loader is not None:
                n_val = len(validation_loader)
                with torch.no_grad():
                    L_val = self.validate_epoch(x, validation_loader)
                L_valout = '{0:6.4e}'.format(L_val/n_val)
            else:
                L_valout = "no validation"
            logmsg  = "Epoch {0:5d}:   Train Loss = {1:6.4e}".format(epoch, L_avg)
            logmsg += "   |   Validation Loss = "+L_valout
            print(logmsg)
            for sched in lr_sched: sched.step()
            self.minimize_log.append(L_avg)
        return x, L_avg
        
    
    def train_epoch(self, x, dataloader):
        Ltot = 0.
        for (inputs, refs, weights) in dataloader:
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
            def closure():
                self.opt.zero_grad()
                res = self(x, *inputs)
                loss = self.loss_func(res, refs, nAtoms=nAtoms, 
                            weights=weights, include=self.include_loss,
                            extensive=self.is_extensive)
                loss.backward()
                return loss
            L = self.opt.step(closure)
            Ltot += L.item()
        return Ltot
        
    
    def validate_epoch(self, x, validation_loader):
        Lval = 0.
        for (inputs, refs, weights) in validation_loader:
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
            res = self(x, *inputs)
            loss = self.loss_func(res, refs, nAtoms=nAtoms, 
                            weights=weights, include=self.include_loss,
                            extensive=self.is_extensive)
            Lval += loss.item()
        return Lval
        
    
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
                    msg  = "Unknown scheduler '"+s+"'. Currently, only "
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
        
    
    def process_reference(self, prop, ref):
        if ref is None: return None
        if prop == "forces":
            ref_proc = prepare_array(ref, 'reference forces')
        if torch.is_tensor(ref):
            ref_proc = ref
        elif type(ref) in [int, float]:
            ref_proc = torch.tensor([ref])
        elif type(ref) == list:
            ref_proc = torch.tensor(ref)
        else:
            raise ValueError("Unrecognized type of reference "+prop)
        self.include_loss.append(prop2index[prop])
        return ref_proc

    
    def gradient(self, x):
        """ Return gradient of loss w.r.t. input x at x. """
        L = self(x)
        dLdx = agrad(L, x, retain_graph=True)[0]
        return dLdx
   
