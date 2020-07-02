import torch
import src.models as models
from src.resnets import resnet34, resnet34_sparse
from src.functional import save, load, plot_curves, plot_grads, plot_mags, rgetattr, get_dataloaders, hybrid_grad
import os
import numpy as np
from src.granular import HierarchicalRingTopK, HierarchicalRingOMP
from time import sleep


DEVICE = "cuda"
SAVEFREQ = 1

def run(depth, augmentation, mparams, position, fsmult, kdiv, auxweight, loadmodel, usecase, prefix):


    """###########################################

    torch.manual_seed(1)
    x = torch.randn((128, 3, 32, 32)).cuda()
    ring = models.Conv6_SparseFirst_Hierarchical(4, 2, "regularize").cuda()
    for logits, preds, layer_aux_loss in ring(x):
        print("Done!")
        sleep(5)
        exit(0)
    exit(0)
    ###########################################"""


    torch.autograd.set_detect_anomaly(True)
    #torch.manual_seed(0)

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
            MODELTYPE = models.Conv6
        elif depth == 12:
            MODELTYPE = models.Conv12
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
            elif position == "0_Res":
                MODELTYPE = models.Conv12_Sparse0_Res
            elif position == "01_Res":
                MODELTYPE = models.Conv12_Sparse01_Res
            elif position == "012_Res":
                MODELTYPE = models.Conv12_Sparse012_Res
            elif position == "0123_Res":
                MODELTYPE = models.Conv12_Sparse0123_Res
            elif position == "01234_Res":
                MODELTYPE = models.Conv12_Sparse01234_Res
            elif position == "012345_Res":
                MODELTYPE = models.Conv12_Sparse012345_Res
        else:
            if position == "First":
                MODELTYPE = models.Conv6_SparseFirst
            elif position == "Middle":
                MODELTYPE = models.Conv6_SparseMiddle
            elif position == "Last":
                MODELTYPE = models.Conv6_SparseLast
            elif position == "01":
                MODELTYPE = models.Conv6_Sparse01
            elif position == "012":
                MODELTYPE = models.Conv6_Sparse012
            elif position == "0123":
                MODELTYPE = models.Conv6_Sparse0123
            elif position == "01234":
                MODELTYPE = models.Conv6_Sparse01234
            elif position == "012345_NonIterative":
                MODELTYPE = models.Conv6_Sparse012345_NonIterative
            elif position == "012345":
                MODELTYPE = models.Conv6_Sparse012345
            elif position == "012345_ReLU":
                MODELTYPE = models.Conv6_Sparse012345_ReLU
            elif position == "VanillaResnet":
                MODELTYPE = resnet34
            elif position == "Resnet_Sparse":
                MODELTYPE = resnet34_sparse
            elif position == "First_Hierarchical":
                MODELTYPE = models.Conv6_SparseFirst_Hierarchical
            elif position == "01_Hierarchical":
                MODELTYPE = models.Conv6_Sparse01_Hierarchical
            elif position == "012_Hierarchical":
                MODELTYPE = models.Conv6_Sparse012_Hierarchical
            elif position == "0123_Hierarchical":
                MODELTYPE = models.Conv6_Sparse0123_Hierarchical
            elif position == "01234_Hierarchical":
                MODELTYPE = models.Conv6_Sparse01234_Hierarchical
            elif position == "012345_Hierarchical":
                MODELTYPE = models.Conv6_Sparse012345_Hierarchical
            else:
                raise Exception("Unknown position.")
            
        TAG += "_"+position + "_[FS:{}, KD:{}, AuxWgt:{}]_UseCase:{}".format(fsmult, kdiv, auxweight, usecase)

    if not mparams:
        TAG += "_oldParams"
    if not augmentation:
        TAG += "_noAug"

    trainloader, testloader = get_dataloaders(augmentation, BATCH)
    model = MODELTYPE(*MODELARGS).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=NESTEROV, weight_decay=DECAY)

    print("Training [{}]".format(TAG))

    if "NonIterative" in position:
        train_fn = train_epochs
    else:
        train_fn = train_epochs_iterative
    train_fn(EPOCHS, auxweight, model, trainloader, testloader, criterion, optimizer, LRDROPS, LRFACTOR, TAG, usecase, loadmodel=loadmodel)


def train_epochs_iterative(epochs, auxweight, model, trainloader, valloader, criterion, optimizer, lrdrops, lrfactor, tag, usecase, loadmodel=False, train_accs=None, test_accs=None, epochs_trained=None):

    train_accs = [] if train_accs is None else train_accs
    test_accs = [] if test_accs is None else test_accs

    if loadmodel:
        dirpath = "./models/"+tag
        if os.path.isdir(dirpath):
            files = [f for f in os.listdir(dirpath) if ".pt" in f]
            if len(files) > 0:
                filename = sorted(files)[-1]
                model, optimizer, epochs_trained, train_accs, test_accs = load(model, optimizer, dirpath, filename)


    if epochs_trained is not None:
        epoch = epochs_trained
    else:
        epochs_trained = 0
        epoch = 0

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
        

        avg_train_loss, grads, mags = train_epoch_iterative(model, auxweight, trainloader, criterion, optimizer, usecase)
        train_accs.append(avg_train_loss)
        #train_grads += grads
        #train_mags += mags

        avg_loss = evaluate_iterative(model, valloader, criterion)
        test_accs.append(avg_loss)

        print("Avg Acc: Train[{:.2f}], Test[{:.2f}]".format(100*train_accs[-1], 100*test_accs[-1]))

        if epoch % 5 == 0:
            plot_curves(train=train_accs, test=test_accs, path="./visualizations", tag=tag)
        #plot_grads([train_grads], "./visualizations/grads")
        #plot_mags([train_mags], "./visualizations/mags")

        if epoch % SAVEFREQ == 0:
            save(model, optimizer, epoch, train_accs, test_accs, "./models/"+tag)

def train_epoch_iterative(model, auxweight, dataloader, criterion, optimizer, usecase):
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
        # ------- Iterative accumulation of grads, discarding the auxiliary computation graph after each step. --------
        for logits, preds, layer_aux_loss in model(x):
            if logits is None and preds is None and layer_aux_loss is not None:
                loss = layer_aux_loss if usecase == "pretrain" else auxweight*layer_aux_loss
                loss.backward(retain_graph=True)
                layer_aux_loss = None
                loss = None
            elif layer_aux_loss is None and logits is not None and preds is not None:
                acc = torch.sum((preds == targets).float()) / preds.numel()
                classification_loss = criterion(logits, targets)
                loss = (1-auxweight)*classification_loss if usecase == "regularize" else classification_loss
                loss.backward(retain_graph=True)
                accs.append(acc.item())
                layer_aux_loss = None
                loss = None
            elif layer_aux_loss is None and logits is None and preds is None:
                    pass
            else:
                raise Exception("Invalid combination of model outputs.")
        # ---------------------------------------------------------------------------------------------------------------
        optimizer.step()
        
    return torch.mean(torch.tensor(accs)).item(), grads, mags

def evaluate_iterative(model, dataloader, criterion):
    model.eval()
    accs = []
    for i, batch in enumerate(dataloader):
        print("                               \r{:.2f}% done...".format(100*(i+1)/len(dataloader)), end='\r')
        x, targets = batch
        x = x.to(DEVICE)
        targets = targets.to(DEVICE)

        for layer_num, (logits, preds, layer_aux_loss) in enumerate(model(x)):
            if logits is None and preds is None and layer_aux_loss is not None:
                pass

            elif layer_aux_loss is None and logits is not None and preds is not None:
                acc = torch.sum((preds == targets).float()) / preds.numel()
                classification_loss = criterion(logits, targets)

                accs.append(acc.item())
            elif layer_aux_loss is None and logits is None and preds is None:
                    pass
            else:
                raise Exception("Invalid combination of model outputs.")

    return torch.mean(torch.tensor(accs)).item()





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



    if epochs_trained is not None:
        epoch = epochs_trained
    else:
        epochs_trained = 0
        epoch = 0

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
            
            optimizer.zero_grad()
            total_loss.backward()

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

