#############################################################################
# Base class and functionality of wrappers for SEQC calculations            #
#  - NoScheduler: dummy scheduler (simple pass)                             #
#  - AbstractWrapper: abstract base class as template for wrapper modules   #
#                                                                           #
# Current (Feb/20)                                                          #
# TODO: . typing                                                            #
#       . Improvement of GPU performance?                                   #
#       . parallel training (GPU mode: DistributedDataParallel, CPU: MPI?)  #
#       . enable optimization of only select atoms/elements (mask grad)     #
#############################################################################

import torch
import numpy as np
from warnings import warn
from abc import ABC, abstractmethod
from torch.autograd import grad as agrad
from .loss_functions import *

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.has_cuda else "cpu")

# THIS NEEDS TO BE CONSISTENT WITH DATALOADERS!
prop2index = {'atomization':0, 'energy':1, 'forces':2, 'gap':3}


class NoScheduler:
    """ Dummy scheduler class. """
    def __init__(self): pass
    def step(self, metrics=0.): pass


class ABW(torch.nn.Module, ABC):
    """
    Abstract base class for wrappers.
    Implements common features such as basic initialization, loss evaluation,
    adding loss properties, and minimization.
    
    Individual, concrete loss modules have to extend base `__init__`
    and provide a corresponding implementation of `self.run_calculation(x)`.
    """
    def __init__(self, custom_params=None, loss_include=[], 
                 SCFfail_penalty=[1e2,1e-5,1e-6], loss_type="RSSperAtom",
                 loss_args=(), loss_kwargs={}):
        ## initialize parent modules and attributes
        super(ABW, self).__init__()
        self.implemented_properties = ['atomization','energy','forces','gap']
        self.n_impl = len(self.implemented_properties)
        self.is_extensive = [True, True, True, False]
        self.implemented_loss_types = ["RSSperAtom"]
        ## collect attributes from input
        if loss_type not in self.implemented_loss_types:
            raise ValueError("Unknown loss type '"+loss_type+"'.")
        argstr = "(*loss_args, n_implemented_properties=self.n_impl, **loss_kwargs)"
        exec("self.loss_func = " + loss_type + argstr)
        self.SCFfail_penalty = SCFfail_penalty
        if any(prop not in self.implemented_properties for prop in loss_include):
            raise ValueError("Requested property/ies not available.")
        self.include_loss = sorted([prop2index[prop] for prop in loss_include])
        self.custom_params = custom_params
#        self.last_x = 0.
        
    
    @abstractmethod
    def forward(self, x, species, coordinates):
        """
        Abstract method for running calculation and gathering results.
        This has to be implemented by the individual concrete wrappers
        """
        pass
        
    
    def train(self, x, dataloader, n_epochs=4, include=None, optimizer="Adam",
              opt_kwargs={}, opt_mode="stochastic", n_up_thresh=5, up_thresh=1e-4,
              loss_conv=1e-8, loss_step_conv=1e-8, scheduler=None, scheduler_kwargs={},
              validation_loader=None, SCFfail_penalty=[1e2,1e-5,1e-6]):
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
                default: Adam
                NOTES:
                    - stochastic LBFGS, AdamW, Adadelta, SGD, ASGD seem unreliable!
                    - Nadam questionable (requires careful choice of settings)
                    - Adam, single-epoch LBFGS appear to work OK
                    - Adagrad, Rprop seem stable, but veeeery slow
          . opt_kwargs, dict: dictionary of kwargs for optimizer, default: {}
          . opt_mode, str: mode for taking optimization steps
                'stochastic': take step in every batch (stochastic training)
                'full': accumulate loss of all batches, then take step
                default: 'stochastic'
          . n_up_thresh, int: number of consecutive increasing loss to 
                accept during minimization, default: 5
          . up_thresh, float: criterion to decide whether loss increased
                (allows to ignore small increases), default: 1e-4
          . loss_conv, float: convergence criterion for loss, default 1e-8
          . loss_step_conv, float: convergence criterion per epoch. Stop if
                loss changes less than this per epoch, default: 1e-8
          . scheduler, str/list of str: learning rate scheduler(s) from 
                torch.optim.lr_scheduler, default: None
                NOTE: ReduceLROnPlateau seems to give the most/only reliable opt!
          . scheduler_kwargs, {}/list of {}: kwargs for scheduler(s)
          . validation_loader, torch.utils.data.Dataloader: dataloader for validation
                (see seqm.utils.dataloaders)
          . SCFfail_penalty, tuple/list len(3): scaling, width, and offset along step
                direction of Gaussian penalty function for failed SCF convergence
        
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
        if opt_mode == "stochastic":
            self.train_epoch = self.train_stochastic
        elif opt_mode == "full":
            self.train_epoch = self.train_full
        else:
            raise ValueError("Unknown optimization mode '"+opt_mode+"'")
        self.SCFfail_penalty = SCFfail_penalty
        if any(prop not in self.implemented_properties for prop in include):
            raise ValueError("Requested property/ies not available.")
        if include is not None:
            self.include_loss = sorted([prop2index[prop] for prop in include])
        Lbak, n_up, self.minimize_log = torch.inf, 0, []
        logmsg  = "\n  SEQC OPTIMIZATION BROUGHT TO YOU BY SLOWCODE, INC."
        logmsg += "\n"+"-"*73
        print(logmsg)
#        self.last_x = x.detach().clone()
        for epoch in range(n_epochs):
            if validation_loader is not None:
                x.requires_grad_(False)
                L_val = self.validate_epoch(x, validation_loader)
                L_valout = '{0:6.4e}'.format(L_val)
                x.requires_grad_(True)
            else:
                L_valout = "no validation"
            L_epoch = self.train_epoch(x, dataloader)
            if not np.isfinite(L_epoch):
                raise RuntimeError("Current loss (",L_epoch,") is not finite!")
            logmsg  = "Epoch {0:5d}:   Train Loss = {1:6.4e}".format(epoch, L_epoch)
            logmsg += "   |   Validation Loss = "+L_valout
            print(logmsg)
            if L_epoch < loss_conv:
                print("Reached convergence criterion {0:3.2e}.".format(loss_conv))
                break
            DeltaL = L_epoch - Lbak
            if abs(DeltaL) < loss_step_conv:
                print("Loss changed less than {0:3.2e} in epoch.".format(loss_step_conv))
                break
            n_up = int(DeltaL > up_thresh) * (n_up + 1)
            if n_up > n_up_thresh:
                msg  = "Loss increased by {0:3.1e} more than ".format(up_thresh)
                msg += str(n_up_thresh)+" times. This may indicate a failure in"
                msg += " training. You can adjust this limit via 'up_thresh' "
                msg += "and 'n_up_thresh'."
                raise RuntimeError(msg)
            Lbak = L_epoch
            for sched in lr_sched: sched.step(metrics=L_epoch)
            self.minimize_log.append(L_epoch)
        L_end = self.get_loss(x, dataloader)
        self.minimize_log.append(L_end)
        if validation_loader is not None:
            x.requires_grad_(False)
            L_val = self.validate_epoch(x, validation_loader)
            L_valout = '{0:6.4e}'.format(L_val)
            x.requires_grad_(True)
        else:
            L_valout = "no validation"
        logmsg  = "-"*73+"\n"
        logmsg += "Final:         Train Loss = {0:6.4e}".format(L_end)
        logmsg += "   |   Validation Loss = "+L_valout
        print(logmsg)
        return x, L_end
        
    
    def train_stochastic(self, x, dataloader):
        Ltot, self.ntot_tst = 0., 0.
        self.loss_func.raw_loss = 0.
        self.loss_func.individual_loss[:] = 0.
        for (inputs, refs, weights) in dataloader:
            inputs = [inp.to(device) for inp in inputs]
            refs = [ref.to(device) for ref in refs]
            weights = [w.to(device) for w in weights]
            self.ntot_tst += inputs[0].shape[0]
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
#            last_step = x.detach().clone() - self.last_x
            def closure():
                self.opt.zero_grad()
                res, failed = self(x, *inputs)
                if failed.any():
                    mask = torch.where(~failed)
                    res = [r[mask] for r in res]
                    my_w = [w[mask] for w in weights]
                    my_refs = [ref[mask] for ref in refs]
                    my_nA = nAtoms[mask]
                    self.ntot_tst -= failed.count_nonzero()
#                    with torch.no_grad():
#                        penalty = failed.count_nonzero()*self.SCFfail_penalty[0]
#                        penalty_width = self.SCFfail_penalty[1]
#                        offset = last_step * self.SCFfail_penalty[2]
#                        center = x.detach().clone() + offset
#                        self.loss_func.add_penalty(center,sigma=penalty_width,
#                                                   scale=penalty)
                else:
                    my_refs, my_w, my_nA = refs, weights, nAtoms
                loss = self.loss_func(res, my_refs, x, nAtoms=my_nA, 
                                    weights=my_w, include=self.include_loss,
                                    extensive=self.is_extensive)
                x.grad = agrad(loss, x)[0]
                return loss
#            self.last_x = x.detach().clone()
            L = self.opt.step(closure)
            Ltot += L.item()
        self.ntot_tst = max(self.ntot_tst, 1)
        self.raw_loss = self.loss_func.raw_loss / self.ntot_tst
        self.individual_loss = self.loss_func.individual_loss / self.ntot_tst
        return Ltot / self.ntot_tst
        
    
    def train_full(self, x, dataloader):
        self.loss_func.raw_loss = 0.
        self.loss_func.individual_loss[:] = 0.
        def closure():
            ntot, Ltot = 0., torch.tensor(0., device=device)
            self.opt.zero_grad()
            dLdx = torch.zeros_like(x, requires_grad=False, device=device)
            for (inputs, refs, weights) in dataloader:
                inputs = [inp.to(device) for inp in inputs]
                refs = [ref.to(device) for ref in refs]
                weights = [w.to(device) for w in weights]
                ntot += inputs[0].shape[0]
                nAtoms = torch.count_nonzero(inputs[0], dim=1)
#                last_step = x.detach().clone() - self.last_x
                res, failed = self(x, *inputs)
                if failed.any():
                    mask = torch.where(~failed)
                    res = [r[mask] for r in res]
                    refs = [ref[mask] for ref in refs]
                    weights = [w[mask] for w in weights]
                    nAtoms = nAtoms[mask]
                    ntot -= failed.count_nonzero()
#                    with torch.no_grad():
#                        penalty = failed.count_nonzero()*self.SCFfail_penalty[0]
#                        penalty_width = self.SCFfail_penalty[1]
#                        offset = last_step * self.SCFfail_penalty[2]
#                        center = x.detach().clone() + offset
#                        self.loss_func.add_penalty(center,sigma=penalty_width,
#                                                   scale=penalty)
                loss = self.loss_func(res, refs, x, nAtoms=nAtoms,
                            weights=weights, include=self.include_loss,
                            extensive=self.is_extensive)
                dLdx = dLdx + agrad(loss, x)[0]
                Ltot = Ltot + loss.item()
            ntot = max(ntot, 1)
            self.raw_loss = self.loss_func.raw_loss / ntot
            self.individual_loss = self.loss_func.individual_loss / ntot
            Ltot = Ltot / ntot
#            Ltot.backward()
            x.grad = dLdx
            return Ltot
#        self.last_x = x.detach().clone()
        L = self.opt.step(closure)
        Lout = L.item()
        return Lout
        
    
    def validate_epoch(self, x, validation_loader):
        x.requires_grad_(False)
        Lval, nval = 0., 0.
        for (inputs, refs, weights) in validation_loader:
            inputs = [inp.to(device) for inp in inputs]
            refs = [ref.to(device) for ref in refs]
            weights = [w.to(device) for w in weights]
            nval += inputs[0].shape[0]
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
            res, failed = self(x, *inputs)
            if failed.any():
                mask = torch.where(~failed)
                res = [r[mask] for r in res]
                refs = [ref[mask] for ref in refs]
                weights = [w[mask] for w in weights]
                nAtoms = nAtoms[mask]
                nval -= failed.count_nonzero().item()
            loss = self.loss_func(res, refs, x, nAtoms=nAtoms, 
                            weights=weights, include=self.include_loss,
                            extensive=self.is_extensive)
            Lval += loss.item()
        nval = max(nval, 1)
        return Lval / nval
        
    
    def get_loss(self, x, dataloader, k_fail=False):
        x.requires_grad_(False)
        self.loss_func.raw_loss = 0.
        self.loss_func.individual_loss[:] = 0.
        self.loss_func.individual_per_mol = [[],]*self.n_impl
        Ltot, ntot, nfail_tot = 0., 0., 0
        for (inputs, refs, weights) in dataloader:
            inputs = [inp.to(device) for inp in inputs]
            refs = [ref.to(device) for ref in refs]
            weights = [w.to(device) for w in weights]
            ntot += inputs[0].shape[0]
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
            res, failed = self(x, *inputs)
            if failed.any():
                mask = torch.where(~failed)
                res = [r[mask] for r in res]
                refs = [ref[mask] for ref in refs]
                weights = [w[mask] for w in weights]
                nAtoms = nAtoms[mask]
                n_fail = failed.count_nonzero()
                nfail_tot += n_fail.item()
                ntot -= n_fail.item()
#                with torch.no_grad():
#                    penalty = n_fail * self.SCFfail_penalty[0]
#                    penalty_width = self.SCFfail_penalty[1]
#                    center = x.detach().clone()
#                    self.loss_func.add_penalty(center,sigma=penalty_width,
#                                               scale=penalty)
            L = self.loss_func(res, refs, x, nAtoms=nAtoms,
                               weights=weights, include=self.include_loss,
                               extensive=self.is_extensive)
            Ltot += L.item()
        ntot = max(ntot, 1)
        Ltot = Ltot / ntot
        self.raw_loss = self.loss_func.raw_loss / ntot
        self.individual_loss = self.loss_func.individual_loss / ntot
        self.individual_per_mol = self.loss_func.individual_per_mol
        if (nfail_tot > 0):
            print("SCF failed to converge for ",nfail_tot," molecules.")
            if k_fail: return 1e3 * nfail_tot + self.raw_loss
        return Ltot
        
    
    def loss_and_grad(self, x, dataloader, k_fail=False):
        Ltot, ntot, nfail_tot = 0., 0., 0
        dLdx = torch.zeros_like(x, requires_grad=False, device=device)
        self.loss_func.raw_loss = 0.
        self.loss_func.individual_loss[:] = 0.
        for (inputs, refs, weights) in dataloader:
            inputs = [inp.to(device) for inp in inputs]
            refs = [ref.to(device) for ref in refs]
            weights = [w.to(device) for w in weights]
            ntot += inputs[0].shape[0]
            nAtoms = torch.count_nonzero(inputs[0], dim=1)
            res, failed = self(x, *inputs)
            if failed.any():
                mask = torch.where(~failed)
                res = [r[mask] for r in res]
                refs = [ref[mask] for ref in refs]
                weights = [w[mask] for w in weights]
                nAtoms = nAtoms[mask]
                n_fail = failed.count_nonzero().item()
                nfail_tot += n_fail
                ntot -= n_fail
#                last_step = x.detach().clone() - self.last_x
#                with torch.no_grad():
#                    penalty = n_fail * self.SCFfail_penalty[0]
#                    penalty_width = self.SCFfail_penalty[1]
#                    offset = last_step * self.SCFfail_penalty[2]
#                    center = x.detach().clone() + offset
#                    self.loss_func.add_penalty(center, sigma=penalty_width,
#                                               scale=penalty)
            L = self.loss_func(res, refs, x, nAtoms=nAtoms,
                               weights=weights, include=self.include_loss,
                               extensive=self.is_extensive)
            dLdx = dLdx + agrad(L, x)[0]
            Ltot = Ltot + L.item()
        if nfail_tot > 0: print(nfail_tot," molecules not converged.")
#        self.last_x = x.detach().clone()
        ntot = max(ntot, 1)
        self.raw_loss = self.loss_func.raw_loss / ntot
        self.individual_loss = self.loss_func.individual_loss / ntot
        Lout, dLdx = Ltot / ntot, dLdx / ntot
        if k_fail and (nfail_tot > 0): Lout = 1e3 * nfail_tot + self.raw_loss
        return Lout, dLdx.detach().numpy()
        
    
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
                    sched_inst = my_sched(optimizer, **sched_kwargs[i])
                    if s not in ["ReduceLROnPlateau"]:
                        step_bak = sched_inst.step
                        def step_new(metrics=0., **kwargs):
                            step_bak(**kwargs)
                        sched_inst.step = step_new
                    lr_sched.append(sched_inst)
                except AttributeError:
                    msg  = "Unknown scheduler '"+s+"'. Currently, only "
                    msg += "schedulers in torch.optim.lr_scheduler supported"
                    raise ImportError(msg)
            return lr_sched
        elif isinstance(scheduler, str):
            try:
                my_sched = getattr(torch.optim.lr_scheduler, scheduler)
                sched_inst = my_sched(optimizer, **sched_kwargs)
                if scheduler not in ["ReduceLROnPlateau"]:
                    step_bak = sched_inst.step
                    def step_new(metrics=0., **kwargs):
                        step_bak(**kwargs)
                    sched_inst.step = step_new
                return [sched_inst]
            except AttributeError:
                msg  = "Unknown scheduler '"+scheduler+"'. Currently, only "
                msg += "schedulers in torch.optim.lr_scheduler supported"
                raise ImportError(msg)
        else:
            msg = "Unrecognized type '"+type(scheduler)+"' for 'scheduler'"
            raise ValueError(msg)
        
    
    def get_raw_loss(self):
        return self.raw_loss
        
    def get_individual_loss(self, prop, per_molecule=False):
        idx = prop2index[prop]
        if per_molecule:
            return self.individual_per_mol[idx]
        else:
            return self.individual_loss[idx].item()

