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


class RSSperAtom(torch.nn.Module):
    """
    Basic implementation of residual sum of squares loss.
    
    Parameters:
    -----------
      . prediction, torch.Tensor: predicted property for each molecule
      . reference, torch.Tensor: corresponding reference value for property
      . x, torch.Tensor: current parameters to be optimized
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
        self.penalties = []
    
    def forward(self, predictions, references, x=0., nAtoms=[], weights=[], 
                extensive=[], include=[], masking=None):
        if masking is None:
            masking = torch.tensor([False,]*predictions[0].shape[0])
        loss_mask = torch.where(masking, 0., 1.)
        loss = torch.tensor(0., requires_grad=True)
        for i in include:
            delta2 = torch.square(predictions[i] - references[i])
            if delta2.dim() > 2:
                sumaxes = tuple(range(predictions[i].dim()))[1:]
                delta2_w = weights[i] * delta2.sum(dim=sumaxes)
            else:
                delta2_w = weights[i] * delta2
            masked_delta2_w = loss_mask * delta2_w
            if extensive[i]:
                my_RSS = (masked_delta2_w / nAtoms).sum()
            else:
                my_RSS = masked_delta2_w.sum()
            loss = loss + my_RSS
        for pen in self.penalties: loss = loss + pen(x)
        return loss
    
    def add_penalty(self, center, sigma=0.01, scale=1.):
        def f_pen(x):
            return scale * torch.exp(-torch.square(x - center).sum() / sigma)
        self.penalties.append(f_pen)
    

