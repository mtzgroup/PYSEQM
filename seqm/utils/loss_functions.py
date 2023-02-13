#############################################################################
# Loss functions to optimize SEQC parametrizations                          #
#  - RSSperAtom: Residual sum of squares weighted by number of atoms for    #
#                extensive properties                                       #
#                                                                           #
# Current (Feb/12)                                                          #
# TODO: . typing                                                            #
#       . enable loss functions from pytroch (at the moment: custom RSS)    #
#############################################################################

import torch


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
    def __init__(self): super(RSSperAtom, self).__init__()
    
    def forward(self, predictions, references, nAtoms=[], weights=[], 
                extensive=[], include=[]):
        loss = torch.tensor(0., requires_grad=True)
        for i in include:
            delta2 = torch.square(predictions[i] - references[i])
            if delta2.dim() > 2:
                sumaxes = tuple(range(predictions[i].dim()))[1:]
                delta2_w = weights[i] * delta2.sum(dim=sumaxes)
            else:
                delta2_w = weights[i] * delta2
            if extensive[i]:
                my_RSS = (delta2_w / nAtoms).sum()
            else:
                my_RSS = delta2_w.sum()
            loss = loss + my_RSS
        return loss
