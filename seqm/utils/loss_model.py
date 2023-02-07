import torch
from warnings import warn
from abc import ABC, abstractmethod
from torch.autograd import grad as agrad
from .pyseqm_helpers import prepare_array, Orderator
from .seqm_runners import SEQM_multirun_core
from .kernel_model import AMASE_multirun_core


torch.set_default_dtype(torch.float64)
prop2index = {'atomization':0, 'energy':1, 'forces':2, 'gap':3}


class AbstractLoss(torch.nn.Module, ABC):
    """
    Abstract base class for loss modules.
    Implements common features such as basic initialization, loss evaluation,
    adding loss properties, and minimization.
    
    Individual, concrete loss modules have to extend base `__init__`
    and provide a corresponding implementation of `self.run_calculation(x)`.
    """
    def __init__(self, species, coordinates, custom_params=None):
        super(AbstractLoss, self).__init__()
        ## initialize parent module and attributes
        self.implemented = ['energy', 'forces', 'gap', 'atomization']
        self.n_implemented = len(self.implemented)
        self.include = [False,]*self.n_implemented
        self.weights = torch.zeros(self.n_implemented)
        
        ## collect attributes from input
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
        if self.include[0]:
            DeltaA2 = torch.square(res[0] - self.atomization_ref)
            Deltas[0] = (DeltaA2 / self.nAtoms).sum()
        if self.include[1]:
            DeltaE2 = torch.square(res[1] - self.energy_ref)
            Deltas[1] = (DeltaE2 / self.nAtoms).sum()
        if self.include[2]:
            DeltaF2 = torch.square(res[2] - self.forces_ref).sum(dim=(1,2))
            Deltas[2] = (DeltaF2 / self.nAtoms).sum()
        if self.include[3]:
            DeltaG2 = torch.square(res[3] - self.gap_ref)
            Deltas[3] = DeltaG2.sum()
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
        if prop not in self.implemented:
            msg  = "Only '"+"', '".join(self.implemented_properties)
            msg += "' implemented for loss. Check for typos or write "
            msg += "coresponding loss function for '"+prop+"'."
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
        self.include[prop2index[prop]] = True
    
    def minimize(self, x_init, options):
        """ Generic minimization routine. """
        #TODO: minimize(self.forward, x_init, options)
        raise NotImplementedError
    
    def gradient(self, x):
        """ Return gradient of loss at input x. """
        L = self.forward(x)
        dLdx = agrad(L, x, retain_graph=True)[0]
        return dLdx
        
    

#############################################
##          CONCRETE LOSS MODULES          ##
#############################################

class SEQM_Loss(AbstractLoss):
    def __init__(self, species, coordinates, custom_params=None, 
                 seqm_settings=None):
        ## initialize parent module
        super(SEQM_Loss, self).__init__(species, coordinates, 
                                        custom_params=custom_params)
        self.runner = SEQM_multirun_core(self.species, self.coordinates,
            custom_params=self.custom_params, seqm_settings=seqm_settings)
    
    def run_calculation(self, p):
        return self.runner(p)
        
    

class AMASE_Loss(AbstractLoss):
    def __init__(self, species, coordinates, desc, reference_Z, 
        reference_desc, reference_coordinates=None, custom_params=None, 
        seqm_settings=None, mode="full", custom_reference=None, expK=1):
        ## initialize parent module
        super(AMASE_Loss, self).__init__(species, coordinates,
                                         custom_params=custom_params)
        Z_ref = prepare_array(reference_Z, "atomic numbers")
        self.runner = AMASE_multirun_core(self.species, desc, 
                self.coordinates, Z_ref, reference_desc,
                reference_coordinates=reference_coordinates, 
                seqm_settings=seqm_settings, mode=mode, 
                custom_params=custom_params, expK=expK, 
                custom_reference=custom_reference)
    
    def run_calculation(self, A):
        return self.runner(A)
        
    

