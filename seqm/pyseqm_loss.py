from warnings import warn
msg  = "Importing old module for loss evaluation in pyseqm, available "
msg += "for backward compatibility. This will be removed in the next "
msg += "minor version. Please move to seqm.utils.loss_model."
warn(msg, DeprecationWarning)
from .utils.pyseqm_loss_old import *
