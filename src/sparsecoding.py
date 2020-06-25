import torch
import torch.nn as nn


"""
class SparseConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, bias=True)
    
    def 

"""







class OMPOptimizer(torch.optim.SGD):
    def __init__(self, model, lr, momentum, nesterov, weight_decay):
        super().__init__(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        self.model = model
    
    def step(self):
        super().step()

        recalculate_interactions = getattr(self.model, "recalculate_atom_interactions", None)
        if callable(recalculate_interactions):
            recalculate_interactions()


