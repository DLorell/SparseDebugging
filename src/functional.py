import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import io
import copy
import os
import functools

def batch_omp(activations, D, k):
    device = activations.device

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


def get_dataloaders(augment, batch_size):
    if augment:
        train_transform = transforms.Compose([
                                    transforms.Pad(padding=(4, 4, 4, 4)),
                                    transforms.RandomCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        train_transform = transforms.Compose([
                                    #transforms.Pad(padding=(4, 4, 4, 4)),
                                    #transforms.RandomCrop(32),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
    trainset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
    testset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader


def save(model, optimizer, epochs_trained, train_accs, test_accs, dirpath):
    state = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epochs_trained": epochs_trained,
        "train_accs": train_accs,
        "test_accs": test_accs
    }

    savepath = os.path.join(dirpath, str(epochs_trained)+".pt") 
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

    other_saves = [f for f in os.listdir(dirpath) if ".pt" in f]
    for save in other_saves:
        othersavepath = os.path.join(dirpath, save)
        os.remove(othersavepath)

    torch.save(state, savepath)

def load(model, optimizer, dirpath, filename):
    save = os.path.join(dirpath, filename)
    state = torch.load(dirpath + "/" + filename)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optim_state"])
    return model, optimizer, state["epochs_trained"], state["train_accs"], state["test_accs"]


def rgetattr(obj, path: str, *default):
    """
    :param obj: Object
    :param path: 'attr1.attr2.etc'
    :param default: Optional default value, at any point in the path
    :return: obj.attr1.attr2.etc
    """
    attrs = path.split(".")
    try:
        return functools.reduce(getattr, attrs, obj)
    except AttributeError:
        if default:
            return default[0]
        raise

def plot_curves(train, test, path, tag):
    plt.clf()
    plt.title("Accuracy over Epochs")

    while len(train) < len(test):
        train = train[0] + train
    while len(test) < len(train):
        test = test[0] + test

    domain = list(range(len(train)))

    plt.plot(domain, np.array(train), '-', label="Train: {:.4f}".format(train[-1]))
    plt.plot(domain, np.array(test), "-", label="Test: {:.4f}".format(test[-1]))
    plt.xlabel('Epoch')
    plt.legend(loc="best")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = copy.deepcopy(Image.open(buf))
    buf.close()

    pil_img.save(path + "/acc_curves_"+tag+".png")

def plot_grads(grads, path, tag):
    plt.clf()
    plt.title("Total Gradient Magnitude over Problematic Epoch")


    for i, grad_list in enumerate(grads):
        plt.plot(np.array(grad_list), '-', label="Grad_{}: {:.4f}".format(i, grad_list[-1]))

    plt.xlabel('Iteration')
    plt.legend(loc="best")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = copy.deepcopy(Image.open(buf))
    buf.close()

    pil_img.save(path + "/grad_curves_"+tag+".png")

def plot_mags(grads, path, tag):
    plt.clf()
    plt.title("Total Parameter Magnitude over Problematic Epoch")


    for i, grad_list in enumerate(grads):
        plt.plot(np.array(grad_list), '-', label="2Norm: {:.4f}".format( grad_list[-1]))

    plt.xlabel('Iteration')
    plt.legend(loc="best")

    plt.ylim(bottom=0)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = copy.deepcopy(Image.open(buf))
    buf.close()

    pil_img.save(path + "/mag_curves_"+tag+".png")

def hybrid_grad(model, optimizer, classification_loss, aux_loss):
    reducer_param_names = [key for key in model.state_dict() if "reducer" in key]
    classification_head_param_names = [key for key in model.state_dict() if "classify" in key]

    classification_loss.backward(retain_graph=True)

    c_head_grads = {param_name: rgetattr(model, param_name).grad.clone() if rgetattr(model, param_name).grad is not None else None for param_name in classification_head_param_names}
    reducer_grads = {param_name: rgetattr(model, param_name).grad.clone() if rgetattr(model, param_name).grad is not None else None for param_name in reducer_param_names}

    optimizer.zero_grad()
    aux_loss.backward()

    for reducer in reducer_param_names:
        rgetattr(model, reducer).grad = reducer_grads[reducer]
    
    for c_head in classification_head_param_names:
        rgetattr(model, c_head).grad = c_head_grads[c_head]


def orthonormalize_init(conv):
    nn.init.orthogonal_(conv.weight)
    norms = torch.norm(conv.weight.view(conv.weight.shape[0], -1), p='fro', dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    conv.weight[:] = conv.weight / norms
