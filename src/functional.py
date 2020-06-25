import torch
import torch.nn as nn


def batch_omp(activations, D, k):
    device = activations.device()

    alpha_0, shape = _batch_vectorize(activations)
    G = torch.mm(D.T, D)
    I = None
    L = torch.ones((alpha_0.shape[0], 1)).to(device)
    alpha = alpha_0
    s = torch.zeros_like(alpha)
    n = 1
    
    while _k_criterion(s, k):
        # 5
        _, k_hat = torch.max(torch.abs(alpha), dim=1)
        k_hat = k_hat.unsqueeze(1)
        #6
        if n > 1:
        #7
            if n == 2:
                w = _batched_index2d(G, I, k_hat)
            else:
                w = torch.bmm(torch.inverse(L), _batched_index2d(G, I, k_hat))
        #8
            w_t = w.permute(0, 2, 1)
            sqrt = torch.sqrt(1 - torch.bmm(w_t, w))

            if n == 2:
                zeros = torch.zeros((L.shape[0])).float().to(device)
                L = torch.stack((L.squeeze(-1).squeeze(-1), zeros, w_t.squeeze(-1).squeeze(-1), sqrt.squeeze(-1).squeeze(-1)), dim=1).view(-1, 2, 2)
            else:
                zeros = torch.zeros((L.shape[0], L.shape[1]+w_t.shape[1]-sqrt.shape[1], 1)).to(device)
                L_1 = torch.cat((L, w_t), dim=1)
                L_2 = torch.cat((zeros, sqrt), dim=1)
                L = torch.cat((L_1, L_2), dim=2)
        #10
        I = torch.cat((I, k_hat), dim=1) if I is not None else k_hat
        #11
        if n == 1:
            s.scatter_(1, I, torch.gather(alpha_0, 1, I))
        else:
            inv = torch.inverse(torch.bmm(L, L.permute(0, 2, 1)))
            c = torch.bmm(inv, torch.gather(alpha_0, 1, I).unsqueeze(-1)).squeeze(-1)
            s.scatter_(1, I, c)
        #12
        beta = torch.bmm(_batched_index2d_justcol(G, I), torch.gather(s, 1, I).unsqueeze(-1))
        #13
        alpha = alpha_0 - beta.squeeze(-1)
        #16
        n += 1
        
        """
        print("Current max k: {}, Current min k: {}, Current avg k: {}".format(
            torch.max(torch.sum((s != 0).float(), dim=1)),
            torch.min(torch.sum((s != 0).float(), dim=1)),
            torch.mean(torch.sum((s != 0).float(), dim=1))))
        """
    
    return _batch_unvectorize(s, shape)

def _k_criterion(s, k):
    return torch.max(torch.sum((s != 0).float(), dim=1)) < k
    
def _batched_index2d(M, I, J):
    # I is rows, J is cols.
    assert len(J.shape) == len(I.shape) == 2
    batch = J.shape[0]
    
    if len(M.shape) != 3:
        M = torch.stack([M for i in range(batch)], dim=0)
    
    J_plus = torch.stack([J for i in range(M.shape[-1])], dim=1)
    I_plus = torch.stack([I for i in range(J.shape[-1])], dim=2)
    
    indexed = torch.gather(torch.gather(M, 2, J_plus), 1, I_plus)
    return indexed

def _batched_index2d_justcol(M, I):
    # I is cols, J is rows.
    assert len(I.shape)  == 2
    batch = I.shape[0]
    
    if len(M.shape) != 3:
        M = torch.stack([M for i in range(batch)], dim=0)
    
    I_plus = torch.stack([I for i in range(M.shape[-1])], dim=1)
    
    indexed = torch.gather(M, 2, I_plus)
    return indexed

def _batch_vectorize(X):
    X = X.permute(0, 2, 3, 1).contiguous()
    (a, b, c) = X.shape[:-1] 
    return X.view(-1, X.shape[-1]), (a,b,c)

def _batch_unvectorize(X, shape):
    a,b,c = shape
    X = X.view(a, b, c, X.shape[-1])
    return X.permute(0, 3, 1, 2).contiguous()