import torch.nn as nn
import torch

class Print(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class Linearize(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)



class Conv12(nn.Module):
    def __init__(self, usecase):
        super().__init__()
        self.usecase = usecase

        self.layer0_0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))


        self.layer1_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))


        self.layer2_0 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0))
        self.pad2 = nn.ReflectionPad2d(1)
        self.layer2_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0))


        self.layer3_0 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=0))
        self.pad3 = nn.ReflectionPad2d(1)
        self.layer3_1 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=0))


        self.layer4_0 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 192, kernel_size=3, stride=1, padding=0))
        self.pad4 = nn.ReflectionPad2d(1)
        self.layer4_1 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))
        

        self.layer5_0 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0))
        self.pad5 = nn.ReflectionPad2d(1)
        self.layer5_1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0))

        self.classify_bnorm = nn.BatchNorm2d(256)
        self.classify_fc = nn.Linear(256*2*2, 100)


    def forward(self, x):
        x = self.layer0_0(x)
        x = self.pad0(x)
        x = self.layer0_1(x)

        x = self.layer1_0(x)
        x = self.pad1(x)
        x = self.layer1_1(x)

        x = self.layer2_0(x)
        x = self.pad2(x)
        x = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.pad3(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.pad4(x)
        x = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)


        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, None

class Conv6(nn.Module):
    def __init__(self, usecase):
        super().__init__()
        self.usecase = usecase

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0))

        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))


        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0))


        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=0))


        self.layer4 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))
        

        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0))

        self.classify_bnorm = nn.BatchNorm2d(256)
        self.classify_fc = nn.Linear(256*2*2, 100)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = self.layer5(x)

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, None


class TopK(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, k):
        top_vals, top_idxs = torch.topk(torch.abs(x), k, dim=1, sorted=False)
        out = torch.zeros_like(x) + -999.99
        out.scatter_(dim=1, index=top_idxs, src=top_vals)
        mask = (out != -999.99)
        out = x * mask
        return out

class SparseCodingLayer_First(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__()
        self.kernel_size = 3
        self.k = k
        self.padding = padding
        self.stride = stride

        self.encoder = nn.Conv2d(in_dim, filterset_size, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        #self.relu = nn.ReLU()
        self.topk = TopK()


        self.decoder = nn.Conv2d(filterset_size, in_dim*self.kernel_size*self.kernel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reducer = nn.Conv2d(filterset_size, out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()

    def forward(self, x):
        aux = self.encoder(x)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)
        aux = self.sigmoid(aux)
        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        #x = self.relu(x)
        x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        if self.padding > 0:
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+self.padding, x.shape[3]+self.padding))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for h in range(bleed, x.shape[-2] - bleed, self.stride):
            for w in range(bleed, x.shape[-1] - bleed, self.stride):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)
                out[:, :, h-bleed, w-bleed] = patch
        return out

class SparseCodingLayer_AfterConv(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__()
        self.kernel_size = 3
        self.k = k
        self.padding = padding
        self.stride = stride

        self.bn = nn.BatchNorm2d(in_dim)
        self.aux_bn = nn.BatchNorm2d(in_dim*self.kernel_size*self.kernel_size)

        self.encoder = nn.Conv2d(in_dim, filterset_size, kernel_size=self.kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.topk = TopK()


        self.decoder = nn.Conv2d(filterset_size, in_dim*self.kernel_size*self.kernel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reducer = nn.Conv2d(filterset_size, out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()

    def forward(self, x):
        #x = self.bn(x)
        x = self.relu(x)

        aux = x.detach().clone()
        aux.requires_grad = True

        aux = self.encoder(aux)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        #aux = self.aux_bn(aux)
        aux = self.relu(aux)

        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        #x = self.relu(x)
        x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        if self.padding > 0:
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+self.padding, x.shape[3]+self.padding))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for h in range(bleed, x.shape[-2] - bleed, self.stride):
            for w in range(bleed, x.shape[-1] - bleed, self.stride):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)
                out[:, :, h-bleed, w-bleed] = patch
        return out

class SparseCodingLayer_AfterSparse(SparseCodingLayer_AfterConv):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, k, padding, stride)

    def forward(self, x):
        x = self.bn(x)
        #x = self.relu(x)

        aux = x.detach().clone()
        aux.requires_grad = True

        aux = self.encoder(aux)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        aux = self.aux_bn(aux)
        #aux = self.relu(aux)

        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        #x = self.relu(x)
        x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss




class CustomMaxPool(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.pool = nn.MaxPool2d(factor)
    
    def forward(self, x):

        if isinstance(x, type((1,2))):
            extra = x[1]
            x = x[0]
        else:
            extra = None

        x = self.pool(x)
        if extra is not None:
            return x, extra
        else:
            return x

class Conv6_SparseFirst(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        out_dim = 64
        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=out_dim, filterset_size=round(int(out_dim*filter_set_mult)), k=round(int(out_dim/k_div)))

    def forward(self, x):
        x, aux_loss = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None
        
        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_SparseMiddle(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        out_dim = 160
        self.layer3 = SparseCodingLayer_AfterConv(in_dim=128, out_dim=out_dim, filterset_size=round(int(out_dim*filter_set_mult)), k=round(int(out_dim/k_div)))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x, aux_loss = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_c(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_SparseLast(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        out_dim = 256
        self.layer5 = SparseCodingLayer_AfterConv(in_dim=192, out_dim=out_dim, filterset_size=round(int(out_dim*filter_set_mult)), k=round(int(out_dim/k_div)))

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x, aux_loss = self.layer5(x)

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss


class Conv6_Sparse01(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))

    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_Sparse012(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x, aux_loss_2 = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_2, aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_Sparse0123(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))

    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x, aux_loss_2 = self.layer2(x)
        x, aux_loss_3 = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_3, aux_loss_2, aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_Sparse01234(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.layer4 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))

    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x, aux_loss_2 = self.layer2(x)
        x, aux_loss_3 = self.layer3(x)
        x, aux_loss_4 = self.layer4(x)
        x = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_4, aux_loss_3, aux_loss_2, aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv6_Sparse012345(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.layer4 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))
        self.layer5 = SparseCodingLayer_AfterSparse(in_dim=192, out_dim=256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div)))

    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x, aux_loss_2 = self.layer2(x)
        x, aux_loss_3 = self.layer3(x)
        x, aux_loss_4 = self.layer4(x)
        x, aux_loss_5 = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_5,aux_loss_4, aux_loss_3, aux_loss_2, aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss


class Conv12_Sparse0(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))


    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x = self.layer1_0(x)
        x = self.pad1(x)
        x = self.layer1_1(x)

        x = self.layer2_0(x)
        x = self.pad2(x)
        x = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.pad3(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.pad4(x)
        x = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00,aux_loss_01)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv12_Sparse01(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))

        self.layer1_0 = SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div)))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2)
        )

    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x, aux_loss_10 = self.layer1_0(x)
        x = self.pad1(x)
        x, aux_loss_11 = self.layer1_1(x)

        x = self.layer2_0(x)
        x = self.pad2(x)
        x = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.pad3(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.pad4(x)
        x = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
                                           aux_loss_10, aux_loss_11)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv12_Sparse012(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))

        self.layer1_0 = SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div)))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2)
        )

        self.layer2_0 = SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.pad2 = nn.ReflectionPad2d(1)
        self.layer2_1 = SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x, aux_loss_10 = self.layer1_0(x)
        x = self.pad1(x)
        x, aux_loss_11 = self.layer1_1(x)

        x, aux_loss_20 = self.layer2_0(x)
        x = self.pad2(x)
        x, aux_loss_21 = self.layer2_1(x)

        x = self.layer3_0(x)
        x = self.pad3(x)
        x = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.pad4(x)
        x = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
                                           aux_loss_10, aux_loss_11,
                                           aux_loss_20, aux_loss_21)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv12_Sparse0123(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))

        self.layer1_0 = SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div)))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2)
        )

        self.layer2_0 = SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.pad2 = nn.ReflectionPad2d(1)
        self.layer2_1 = SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


        self.layer3_0 = SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.pad3 = nn.ReflectionPad2d(1)
        self.layer3_1 = SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))


    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x, aux_loss_10 = self.layer1_0(x)
        x = self.pad1(x)
        x, aux_loss_11 = self.layer1_1(x)

        x, aux_loss_20 = self.layer2_0(x)
        x = self.pad2(x)
        x, aux_loss_21 = self.layer2_1(x)

        x, aux_loss_30 = self.layer3_0(x)
        x = self.pad3(x)
        x, aux_loss_31 = self.layer3_1(x)

        x = self.layer4_0(x)
        x = self.pad4(x)
        x = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
                                           aux_loss_10, aux_loss_11,
                                           aux_loss_20, aux_loss_21,
                                           aux_loss_30, aux_loss_31)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv12_Sparse01234(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))

        self.layer1_0 = SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div)))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2)
        )

        self.layer2_0 = SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.pad2 = nn.ReflectionPad2d(1)
        self.layer2_1 = SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


        self.layer3_0 = SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.pad3 = nn.ReflectionPad2d(1)
        self.layer3_1 = SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))


        self.layer4_0 = SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div)))
        self.pad4 = nn.ReflectionPad2d(1)
        self.layer4_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2)
        )
        

        self.layer5_0 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0))
        self.pad5 = nn.ReflectionPad2d(1)
        self.layer5_1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0))

        self.classify_bnorm = nn.BatchNorm2d(256)
        self.classify_fc = nn.Linear(256*2*2, 100)


    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x, aux_loss_10 = self.layer1_0(x)
        x = self.pad1(x)
        x, aux_loss_11 = self.layer1_1(x)

        x, aux_loss_20 = self.layer2_0(x)
        x = self.pad2(x)
        x, aux_loss_21 = self.layer2_1(x)

        x, aux_loss_30 = self.layer3_0(x)
        x = self.pad3(x)
        x, aux_loss_31 = self.layer3_1(x)

        x, aux_loss_40 = self.layer4_0(x)
        x = self.pad4(x)
        x, aux_loss_41 = self.layer4_1(x)

        x = self.layer5_0(x)
        x = self.pad5(x)
        features = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
                                           aux_loss_10, aux_loss_11,
                                           aux_loss_20, aux_loss_21,
                                           aux_loss_30, aux_loss_31,
                                           aux_loss_40, aux_loss_41)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss

class Conv12_Sparse012345(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0_0 = SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.pad0 = nn.ReflectionPad2d(1)
        self.layer0_1 = SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))

        self.layer1_0 = SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div)))
        self.pad1 = nn.ReflectionPad2d(1)
        self.layer1_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2)
        )

        self.layer2_0 = SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.pad2 = nn.ReflectionPad2d(1)
        self.layer2_1 = SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


        self.layer3_0 = SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.pad3 = nn.ReflectionPad2d(1)
        self.layer3_1 = SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))


        self.layer4_0 = SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div)))
        self.pad4 = nn.ReflectionPad2d(1)
        self.layer4_1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2)
        )

        self.layer5_0 = SparseCodingLayer_First(192, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div)))
        self.pad5 = nn.ReflectionPad2d(1)
        self.layer5_1 = SparseCodingLayer_AfterSparse(256, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div)))


    def forward(self, x):
        x, aux_loss_00 = self.layer0_0(x)
        x = self.pad0(x)
        x, aux_loss_01 = self.layer0_1(x)

        x, aux_loss_10 = self.layer1_0(x)
        x = self.pad1(x)
        x, aux_loss_11 = self.layer1_1(x)

        x, aux_loss_20 = self.layer2_0(x)
        x = self.pad2(x)
        x, aux_loss_21 = self.layer2_1(x)

        x, aux_loss_30 = self.layer3_0(x)
        x = self.pad3(x)
        x, aux_loss_31 = self.layer3_1(x)

        x, aux_loss_40 = self.layer4_0(x)
        x = self.pad4(x)
        x, aux_loss_41 = self.layer4_1(x)

        x, aux_loss_50 = self.layer5_0(x)
        x = self.pad5(x)
        features, aux_loss_51 = self.layer5_1(x)

        aux_loss = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
                                           aux_loss_10, aux_loss_11,
                                           aux_loss_20, aux_loss_21,
                                           aux_loss_30, aux_loss_31,
                                           aux_loss_40, aux_loss_41,
                                           aux_loss_50, aux_loss_51)))

        if self.usecase == "random" or self.usecase == "pretrain":
            features = features.detach().clone()
            features.requires_grad = True
        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(features)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss



class SparseCodingLayer_First_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, k):
        super().__init__()
        self.kernel_size = 3
        self.k = k

        self.encoder = nn.Conv2d(in_dim, filterset_size, kernel_size=self.kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.topk = TopK()


        self.decoder = nn.Conv2d(filterset_size, in_dim*self.kernel_size*self.kernel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reducer = nn.Conv2d(filterset_size, out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()

    def forward(self, x):
        aux = self.encoder(x)
        aux = self.relu(aux)
        #aux = self.topk(aux, self.k)
        aux = self.decoder(aux)
        aux = self.sigmoid(aux)
        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        x = self.relu(x)
        #x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        for h in range(bleed, x.shape[-2] - bleed):
            for w in range(bleed, x.shape[-1] - bleed):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)
                out[:, :, h-bleed, w-bleed] = patch
        return out

class SparseCodingLayer_AfterConv_ReLU(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, k):
        super().__init__()
        self.kernel_size = 3
        self.k = k

        self.bn = nn.BatchNorm2d(in_dim)
        self.aux_bn = nn.BatchNorm2d(in_dim*self.kernel_size*self.kernel_size)

        self.encoder = nn.Conv2d(in_dim, filterset_size, kernel_size=self.kernel_size, stride=1, padding=0)
        self.relu = nn.ReLU()
        #self.topk = TopK()


        self.decoder = nn.Conv2d(filterset_size, in_dim*self.kernel_size*self.kernel_size, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reducer = nn.Conv2d(filterset_size, out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()

    def forward(self, x):
        #x = self.bn(x)
        x = self.relu(x)

        aux = x.detach().clone()
        aux.requires_grad = True

        aux = self.encoder(aux)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        #aux = self.aux_bn(aux)
        aux = self.relu(aux)

        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        #x = self.relu(x)
        x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        for h in range(bleed, x.shape[-2] - bleed):
            for w in range(bleed, x.shape[-1] - bleed):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)
                out[:, :, h-bleed, w-bleed] = patch
        return out

class SparseCodingLayer_AfterSparse_ReLU(SparseCodingLayer_AfterConv):
    def __init__(self, in_dim, out_dim, filterset_size, k):
        super().__init__(in_dim, out_dim, filterset_size, k)

    def forward(self, x):
        x = self.bn(x)
        #x = self.relu(x)

        aux = x.detach().clone()
        aux.requires_grad = True

        aux = self.encoder(aux)
        aux = self.relu(aux)
        #aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        aux = self.aux_bn(aux)
        #aux = self.relu(aux)

        aux_loss = self.mse(self.embiggen(x.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        x = self.relu(x)
        #x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss

class Conv6_Sparse012345_ReLU(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.layer0 = SparseCodingLayer_First_ReLU(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse_ReLU(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse_ReLU(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse_ReLU(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.layer4 = nn.Sequential(
            SparseCodingLayer_AfterSparse_ReLU(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))
        self.layer5 = SparseCodingLayer_AfterSparse_ReLU(in_dim=192, out_dim=256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div)))

    def forward(self, x):
        x, aux_loss_0 = self.layer0(x)
        x, aux_loss_1 = self.layer1(x)
        x, aux_loss_2 = self.layer2(x)
        x, aux_loss_3 = self.layer3(x)
        x, aux_loss_4 = self.layer4(x)
        x, aux_loss_5 = self.layer5(x)

        aux_loss = torch.mean(torch.stack((aux_loss_5,aux_loss_4, aux_loss_3, aux_loss_2, aux_loss_1, aux_loss_0)))

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        features = self.classify_bnorm(x)
        features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss


