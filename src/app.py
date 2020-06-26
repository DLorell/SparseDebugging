import torch
from src.models import Conv12, Conv6, Conv6_SparseFirst, Conv6_SparseLast, Conv6_SparseMiddle
import src.models as models
from src.resnets import resnet34, resnet34_sparse
from src.functional import save, load, plot_curves, plot_grads, plot_mags, rgetattr, get_dataloaders, hybrid_grad
import os

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
            elif position == "VanillaResnet":
                MODELTYPE = resnet34
            elif position == "Resnet_Sparse":
                MODELTYPE = resnet34_sparse
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

    train_epochs(EPOCHS, auxweight, model, trainloader, testloader, criterion, optimizer, LRDROPS, LRFACTOR, TAG, usecase, loadmodel=loadmodel)

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


