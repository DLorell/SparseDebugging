import torch.autograd
import src.functional as f


class BatchOMP(torch.autograd.Function):
    def __init__(self):
        super().__init__()

        self.D = None
        self.k = None
        self.activations = None

    def forward(self, activations, D, k):
        self.D = D.detach().clone()
        self.k = k
        self.activations = activations.detach().clone()

        return f.batch_omp(activations, D, k)

    def backward(self, grad_out):
        grad_input = grad_out.clone()
        
        raise Exception('Custom backward called! (Need to implement this.)')

        return grad_input