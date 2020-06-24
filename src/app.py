import torchvision
from torchvision import transforms
import torch
from src.models import Conv12, Conv6, Conv6_SparseFirst, Conv6_SparseLast, Conv6_SparseMiddle
import src.models as models
from src.resnets import resnet34, resnet34_sparse
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import io
import copy
import os
import functools
  
DEVICE = "cuda"
SAVEFREQ = 1

def run(depth, augmentation, mparams, position, fsmult, kdiv, auxweight, loadmodel, usecase, prefix):

    EPOCHS = 200 if mparams else 360
    BATCH = 128 if mparams else 256
    DECAY = 5e-4 if mparams else 1e-4
    NESTEROV= True if mparams else False
    LRFACTOR = 5 if mparams else 10
    LRDROPS = [60, 120, 160] if mparams else [120, 240]

    MODELARGS = [usecase]

    TAG="{}Conv{}".format(prefix, depth)

    if position == "None":
        if depth == 6:
            MODELTYPE = Conv6
        elif depth == 12:
            MODELTYPE = Conv12
        else:
            raise Exception("Unknown depth.")
    else:
        MODELARGS = [fsmult, kdiv] + MODELARGS
        if depth == 12:
            if position == "0":
                MODELTYPE = models.Conv12_Sparse0
            elif position == "01":
                MODELTYPE = models.Conv12_Sparse01
            elif position == "012":
                MODELTYPE = models.Conv12_Sparse012
            elif position == "0123":
                MODELTYPE = models.Conv12_Sparse0123
            elif position == "01234":
                MODELTYPE = models.Conv12_Sparse01234
            elif position == "012345":
                MODELTYPE = models.Conv12_Sparse012345
        else:
            if position == "First":
                MODELTYPE = Conv6_SparseFirst
            elif position == "Middle":
                MODELTYPE = Conv6_SparseMiddle
            elif position == "Last":
                MODELTYPE = Conv6_SparseLast
            elif position == "01":
                MODELTYPE = models.Conv6_Sparse01
            elif position == "012":
                MODELTYPE = models.Conv6_Sparse012
            elif position == "0123":
                MODELTYPE = models.Conv6_Sparse0123
            elif position == "01234":
                MODELTYPE = models.Conv6_Sparse01234
            elif position == "012345":
                MODELTYPE = models.Conv6_Sparse012345
            elif position == "012345_ReLU":
                MODELTYPE = models.Conv6_Sparse012345_ReLU
            elif position == "Resnet":
                MODELTYPE = resnet34
            elif position == "Resnet_Sparse":
                MODELTYPE = resnet34_sparse
            else:
                raise Exception("Unknown position.")
            
        TAG += "_Sparse"+position + "_[FS:{}, KD:{}, AuxWgt:{}]_UseCase:{}".format(fsmult, kdiv, auxweight, usecase)

    if not mparams:
        TAG += "_oldParams"
    if not augmentation:
        TAG += "_noAug"

    trainloader, testloader = get_dataloaders(augmentation, BATCH)
    model = MODELTYPE(*MODELARGS).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=NESTEROV, weight_decay=DECAY)

    print("Training [{}]".format(TAG))

    train_epochs(EPOCHS, auxweight, model, trainloader, testloader, criterion, optimizer, LRDROPS, LRFACTOR, TAG, usecase, loadmodel=loadmodel)

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

def train_epochs(epochs, auxweight, model, trainloader, valloader, criterion, optimizer, lrdrops, lrfactor, tag, usecase, loadmodel=False, train_accs=None, test_accs=None, epochs_trained=None):

    train_accs = [] if train_accs is None else train_accs
    test_accs = [] if test_accs is None else test_accs

    if loadmodel:
        dirpath = "./models/"+tag
        if os.path.isdir(dirpath):
            files = [f for f in os.listdir(dirpath) if ".pt" in f]
            if len(files) > 0:
                filename = sorted(files)[-1]
                model, optimizer, epochs_trained, train_accs, test_accs = load(model, optimizer, dirpath, filename)

    """# --------- Revisualization ----------
    train_accs = train_accs[:200]
    test_accs = test_accs[:200]
    plot_curves(train=train_accs, test=test_accs, path="./visualizations", tag=tag)
    exit(0)
    # ------------------------------------"""


    if epochs_trained is not None:
        epoch = epochs_trained
    else:
        epochs_trained = 0
        epoch = 0

    #avg_loss = evaluate(model, valloader, criterion)
    #test_accs.append(avg_loss)
    #train_accs.append(avg_loss)
    #print("Avg Acc: Train[{:.2f}], Test[{:.2f}]".format(100*train_accs[-1], 100*test_accs[-1]))

    for epoch in range(epochs_trained+1, epochs+1):
        
        if epoch in lrdrops:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / lrfactor

        lrs = []
        for g in optimizer.param_groups:
                lrs.append(g['lr'])
        lrs = set(lrs)
        assert len(lrs) == 1
                
        print("                              \rEpoch [{}/{}] (lr: {})".format(epoch, epochs, list(lrs)[0]))
        

        avg_train_loss, grads, mags = train_epoch(model, auxweight, trainloader, criterion, optimizer, usecase)
        train_accs.append(avg_train_loss)
        #train_grads += grads
        #train_mags += mags

        avg_loss = evaluate(model, valloader, criterion)
        test_accs.append(avg_loss)

        print("Avg Acc: Train[{:.2f}], Test[{:.2f}]".format(100*train_accs[-1], 100*test_accs[-1]))

        if epoch % 5 == 0:
            plot_curves(train=train_accs, test=test_accs, path="./visualizations", tag=tag)
        #plot_grads([train_grads], "./visualizations/grads")
        #plot_mags([train_mags], "./visualizations/mags")

        if epoch % SAVEFREQ == 0:
            save(model, optimizer, epoch, train_accs, test_accs, "./models/"+tag)

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

def train_epoch(model, auxweight, dataloader, criterion, optimizer, usecase):
    model.train()
    accs = []
    grads = []
    mags = []
    for i, batch in enumerate(dataloader):
        print("                              \r{:.2f}% done...".format(100*(i+1)/len(dataloader)), end='\r')
        x, targets = batch
        x = x.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        logits, preds, aux_loss = model(x)
        acc = torch.sum((preds == targets).float()) / preds.numel()
        classification_loss = criterion(logits, targets)

        if usecase == "hybrid":
            hybrid_grad(model, optimizer, classification_loss, aux_loss)
        else:
            if usecase == "random" or usecase == "supervise":
                total_loss = classification_loss
            elif usecase == "pretrain":
                total_loss = classification_loss + aux_loss
            elif usecase == "regularize":
                total_loss = ((1-auxweight)*classification_loss + auxweight*aux_loss)
            else:
                raise Exception('Unknown usecase "{}"'.format(usecase))
            
            total_loss.backward()

        """
        with torch.no_grad():
            accumulator = torch.tensor(0).float()
            w_accumulator = torch.tensor(0).float()
            for param in model.parameters():
                w_accumulator += torch.sum(param ** 2)
                accumulator += torch.sum(param.grad ** 2)
            grad_magnitude = torch.sqrt(accumulator)
            weight_magnitude = torch.sqrt(w_accumulator)
            grads.append(grad_magnitude)
            mags.append(weight_magnitude)
        """

        optimizer.step()
        accs.append(acc.item())

    return torch.mean(torch.tensor(accs)).item(), grads, mags

def evaluate(model, dataloader, criterion):
    model.eval()
    accs = []
    for i, batch in enumerate(dataloader):
        print("                               \r{:.2f}% done...".format(100*(i+1)/len(dataloader)), end='\r')
        x, targets = batch
        x = x.to(DEVICE)
        targets = targets.to(DEVICE)
        logits, preds, aux_loss = model(x)
        acc = torch.sum((preds == targets).float()) / preds.numel()
        #loss = criterion(logits, targets)
        accs.append(acc.item())
    return torch.mean(torch.tensor(accs)).item()

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
