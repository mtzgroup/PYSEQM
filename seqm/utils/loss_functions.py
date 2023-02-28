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
    def __init__(self, n_implemented_properties=4, regularizer=0.):
        super(RSSperAtom, self).__init__()
        self.penalties = []
        self.raw_loss = 0.
        self.individual_loss = torch.tensor([0.,]*n_implemented_properties)
        self.reg = regularizer
    
    def forward(self, predictions, references, x=None, nAtoms=[], weights=[], 
                extensive=[], include=[], masking=None):
        if x is None: x = torch.tensor(0., requires_grad=False)
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
            self.individual_loss[i] += my_RSS.item()
            loss = loss + my_RSS
        self.raw_loss += loss.item()
        for pen in self.penalties: loss = loss + pen(x)
        loss = loss + self.reg * torch.square(x).sum()
        return loss
    
    def add_penalty(self, center, sigma=0.01, scale=1.):
        def f_pen(x):
            return scale * torch.exp(-torch.square(x - center).sum() / sigma)
        self.penalties.append(f_pen)
    

