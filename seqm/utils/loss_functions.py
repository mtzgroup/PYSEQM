#############################################################################
# Loss functions to optimize SEQC parametrizations                          #
#  - RSSperAtom: Residual sum of squares weighted by number of atoms for    #
#                extensive properties                                       #
#                                                                           #
# Current (Feb/12)                                                          #
# TODO: . enable access to individual losses                                #
#       . typing                                                            #
#       . enable loss functions from pytorch (at the moment: custom RSS)    #
#############################################################################

import torch

quadratic_default = {'scale':10., 'shift':0.}
power_default = {'scale':10., 'shift':0., 'exponent':4}
well_default = {'scale':2000, 'shift':0.5, 'steep':0.01}

def null_reg(x):
    return 0.

def quadratic_reg(x, scale, shift):
    return scale * (x+shift).square().sum()

def power_reg(x, scale, shift, exponent):
    return scale * (x+shift).pow(exponent).sum()

def well_reg(x, scale, shift, steep):
    n_wall =  1 / ( torch.exp((x+shift) / steep) + 1 )
    p_wall = -1 / ( torch.exp((x-shift) / steep) + 1 )
    return scale * (1 + n_wall + p_wall).sum()

def get_regularizer_fn(regularizer_dict):
    if regularizer_dict['kind'] is None:
        return null_reg
    elif regularizer_dict['type'] == 'quadratic':
        quadratic_default.update(regularizer_dict)
        a, s = quadratic_default['scale'], quadratic_default['shift']
        def f(x): return quadratic_reg(x, a, s)
        return f
    elif regularizer_dict['type'] == 'power':
        power_default.update(regularizer_dict)
        a, s = power_default['scale'], power_default['shift']
        n = power_default['exponent']
        def f(x): return power_reg(x, a, s, n)
        return f
    elif regularizer_dict['type'] == 'well':
        well_default.update(regularizer_dict)
        a, s = well_default['scale'], well_default['shift']
        m = well_default['steep']
        def f(x): return well_reg(x, a, s, m)
        return f
    else:
        raise ValueError("Regularizer '"+regularizer_dict['type']+"' unknown.")
        
    

class RSSperAtom(torch.nn.Module):
    """
    Basic implementation of residual sum of squares loss.
    
    Parameters:
    -----------
      . prediction, torch.Tensor: predicted property for each molecule
      . reference, torch.Tensor: corresponding reference value for property
      . regularizer, dict: settings for regularizer
      . x, torch.Tensor: current parameters for regularizer
      . intensive, bool: whether property is intensive or not
    
    Returns:
    --------
      . residual sum of squares
          if intensive: sum_{i,...} (predicted_{i,...} - reference_{i,...})^2
          else: sum_A (sum_{i,...} (predicted_{A,i,...} 
                                      - reference_{A,i,...})^2) / nAtoms_A
    """
    def __init__(self, n_implemented_properties=4, regularizer={'kind':None}):
        super(RSSperAtom, self).__init__()
        self.penalties = []
        self.raw_loss = 0.
        self.individual_loss = torch.tensor([0.,]*n_implemented_properties)
        self.reg_func = get_regularizer_fn(regularizer)
    
    def forward(self, predictions, references, x=None, nAtoms=[], weights=[], 
                extensive=[], include=[]):
        if x is None: x = torch.tensor(0., requires_grad=False)
        loss = torch.tensor(0.)
        for i in include:
            delta2 = torch.square(predictions[i] - references[i])
            if delta2.dim() > 2:
                sumaxes = tuple(range(predictions[i].dim()))[1:]
                delta2_w = weights[i] * delta2.sum(dim=sumaxes)
            else:
                delta2_w = weights[i] * delta2
            if extensive[i]:
                my_RSS = (delta2_w / nAtoms)
            else:
                my_RSS = delta2_w
            RSS_sum = my_RSS.sum()
            self.individual_loss[i] += RSS_sum.item()
            loss = loss + RSS_sum
        self.raw_loss = loss.clone()
        loss = loss + self.reg_func(x)
        return loss

