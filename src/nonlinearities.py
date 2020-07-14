import torch.autograd
import src.functional as f
from pytorch_memlab import LineProfiler


class BatchOMP(torch.autograd.Function):

    @staticmethod
    def forward(self, activations, D, k):
        self.k = k
        D = D.detach()
        activations = activations.detach()
        
        self.save_for_backward(D, activations)
        with torch.no_grad():
            #print("Doing it for k=", k)
            #out = f.batch_omp(activations, D, k)

            if k == 32:
                print("I'm in!")
                with LineProfiler(f.batch_omp, f.omp_lines_12_thru_13) as prof:
                    print('Doing it.')
                    out = f.batch_omp(activations, D, k)
                    torch.cuda.empty_cache()
                print("Displaying...")
                print(prof.display())
            else:
                print("Doing it for k=", k)
                out = f.batch_omp(activations, D, k)
                torch.cuda.empty_cache()


        return out

    @staticmethod
    def backward(self, grad_out):
        k = self.k
        D, activations = self.saved_tensors
        grad_input = grad_out.clone()
        
        raise Exception('Custom backward called! (Need to implement this.)')

        return grad_input