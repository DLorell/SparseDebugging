import torch.nn as nn
import torch
import numpy as np
from src.granular import HierarchicalRingTopK, HierarchicalRingOMP

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


""" Non Iterative
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
        x = self.layer5_1(x)


        if self.usecase == "random" or self.usecase == "pretrain":
            x = x.detach().clone()
            x.requires_grad = True

        logits, preds = self.classify(x)

        return logits, preds, None
"""

class CustomPad(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
    def forward(self, x):
        if isinstance(x, type((1,2))):
            extra = x[1]
            x = x[0]
        else:
            extra = None

        x = self.pad(x)

        if extra is not None:
            return x, extra
        else:
            return x

class Center_Projection(nn.Module):
    def __init__(self, in_ch, out_ch, edge, pool=1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.edge = edge
        self.pool = nn.MaxPool2d(pool)
    def forward(self, x):
        return self.pool(self.proj(x[:, :, self.edge:-self.edge, self.edge:-self.edge]))

class Conv12(nn.Module):
    def __init__(self, usecase):
        super().__init__()
        self.usecase = usecase

        self.layer0_0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))


        self.layer1_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))


        self.layer2_0 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0))


        self.layer3_0 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=0))


        self.layer4_0 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 192, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))

        self.layer5_0 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer5_1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0))

        self.classify = ClassificationHead(256*2*2, 100)
        self.layers = []
        self.non_aux = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                        self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, layer_aux_loss = layer(x)

            div = self.num_aux_losses 
            layer_aux_loss = layer_aux_loss / div if div > 0 else layer_aux_loss
            if self.usecase == "random" or self.usecase == "supervise":
                layer_aux_loss = None
            preds = logits = None

            yield logits, preds, layer_aux_loss


        for i, layer in enumerate(self.non_aux):
            x = layer(x)

        layer_aux_loss = None
        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True
        logits, preds = self.classify(x)
        yield logits, preds, layer_aux_loss

class Conv12_Resnet(nn.Module):
    def __init__(self, usecase):
        super().__init__()
        self.usecase = usecase

        self.layer0_0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0))

        self.layer1_0 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))
        self.proj1 = Center_Projection(64, 96, 1, 2)


        self.layer2_0 = nn.Sequential(
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0))
        self.proj2 = Center_Projection(96, 128, 1)


        self.layer3_0 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 160, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 160, kernel_size=3, stride=1, padding=0))
        self.proj3 = Center_Projection(128, 160, 1)


        self.layer4_0 = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.Conv2d(160, 192, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2))
        self.proj4 = Center_Projection(160, 192, 1, 2)

        self.layer5_0 = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=0),
            CustomPad(1))
        self.layer5_1 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0))
        self.proj5 = Center_Projection(192, 256, 1)

        self.classify = ClassificationHead(256*2*2, 100)
        self.layers = []
        self.non_aux = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                        self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]
        self.res_projections = [None, "Dummy", None, self.proj1,
                                None, self.proj2, None, self.proj3,
                                None, self.proj4, None, self.proj5]


    def forward(self, x):
        res_x = None
        for i, layer in enumerate(self.layers):
            x, layer_aux_loss = layer(x)

            if self.res_projections[i] is not None:
                if res_x is not None:
                    x += self.res_projections[i](res_x)
                res_x = x


            div = self.num_aux_losses 
            layer_aux_loss = layer_aux_loss / div if div > 0 else layer_aux_loss
            if self.usecase == "random" or self.usecase == "supervise":
                layer_aux_loss = None
            preds = logits = None

            yield logits, preds, layer_aux_loss


        for j, layer in enumerate(self.non_aux):
            x = layer(x)

            if self.res_projections[i+j+1] is not None:
                if res_x is not None:
                    x += self.res_projections[i+j+1](res_x)
                res_x = x

        layer_aux_loss = None
        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True
        logits, preds = self.classify(x)
        yield logits, preds, layer_aux_loss



class Conv6(nn.Module):
    def __init__(self, usecase):
        super().__init__()
        self.usecase = usecase
        self.num_aux_losses = 0

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

        self.classify = ClassificationHead(256*2*2, 100)

        self.layers = []
        self.non_aux = [self.layer0, self.layer1, self.layer2, self.layer3,
                        self.layer4, self.layer5]
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, layer_aux_loss = layer(x)

            #print(layer_aux_loss.item())
            div = self.num_aux_losses 
            layer_aux_loss = layer_aux_loss / div if div > 0 else layer_aux_loss
            if self.usecase == "random" or self.usecase == "supervise":
                layer_aux_loss = None
            preds = logits = None

            yield logits, preds, layer_aux_loss


        for i, layer in enumerate(self.non_aux):
            x = layer(x)

        layer_aux_loss = None
        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True
        logits, preds = self.classify(x)
        yield logits, preds, layer_aux_loss

class ClassificationHead(nn.Module):
    def __init__(self, in_ch, classes):
        super().__init__()
        self.linear = nn.Linear(in_ch, classes)
        self.bn = nn.BatchNorm2d(256)
    def forward(self, x):
        features = self.bn(x)
        features = features.view(features.shape[0], -1)
        logits = self.linear(features)
        _, preds = logits.max(dim=1)
        return logits, preds



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
        self.out_dim = out_dim

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
        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)
        aux = self.sigmoid(aux)
        aux_loss = self.mse(self.embiggen(aux_in, aux).detach(), aux)
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
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*self.padding), x.shape[3]+(2*self.padding)))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for h in range(bleed, x.shape[-2] - bleed, self.stride):
            for w in range(bleed, x.shape[-1] - bleed, self.stride):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)


                #print(h, h-bleed, w, w-bleed, type(out), type(patch),
                #      patch.shape, out[:, :, h-bleed, w-bleed].shape)
                out[:, :, h-bleed, w-bleed] = patch.to(out.device)

        return out

class SparseCodingLayer_AfterConv(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__()
        self.out_dim = out_dim

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
        x = self.bn(x)
        x = self.relu(x)

        aux = x.detach().clone()
        aux.requires_grad = True

        aux = self.encoder(aux)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        aux = self.aux_bn(aux)
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
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*self.padding), x.shape[3]+(2*self.padding)))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for i, h in enumerate(range(bleed, x.shape[-2] - bleed, self.stride)):
            for j, w in enumerate(range(bleed, x.shape[-1] - bleed, self.stride)):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)
                out[:, :, i, j] = patch.to(out.device)
        return out

class SparseCodingLayer_AfterSparse(SparseCodingLayer_AfterConv):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, k, padding, stride)

    def forward(self, x):
        x = self.bn(x)
        #x = self.relu(x)

        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        #aux = self.relu(aux)
        aux = self.topk(aux, self.k)
        aux = self.decoder(aux)

        aux = self.aux_bn(aux)
        #aux = self.relu(aux)

        aux_loss = self.mse(self.embiggen(aux_in.detach(), aux).detach(), aux)
        aux = None


        x = self.encoder(x)
        #x = self.relu(x)
        x = self.topk(x, self.k)
        x = self.reducer(x)

        return x, aux_loss



class SparseCodingLayer_ArchTopK(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, padding, stride, k):
        super().__init__()

        self.k = k

        self.out_dim = out_dim

        self.kernel_size = 3
        self.padding = padding
        self.stride = stride

        levels, atoms_per_level = self.filterset_to_levels(filterset_size)

        self.encoder = HierarchicalRingTopK(in_dim, levels=levels, ks=[self.k for _ in atoms_per_level])
        self.decoder = nn.Conv2d(np.sum(atoms_per_level), in_dim*self.kernel_size*self.kernel_size, kernel_size=1, stride=1, padding=0)
        self.reducer = nn.Conv2d(np.sum(atoms_per_level), out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()
    
    def forward(self, x):
        raise NotImplementedError

    def filterset_to_levels(self, target):
        cur = 1
        atoms = [8]
        while np.sum(atoms) < target:
            new_atoms = atoms[-1]*2
            if np.sum(atoms)+new_atoms < target:
                cur += 1
                atoms.append(new_atoms)
                continue
            else:
                diff_below = np.abs(np.sum(atoms) - target)
                diff_above = np.abs(np.sum(atoms) + new_atoms - target)

                if diff_below <= diff_above:
                    break
                else:
                    cur += 1
                    atoms.append(new_atoms)
                    break
        return cur, atoms

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        if self.padding > 0:
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*self.padding), x.shape[3]+(2*self.padding)))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for h in range(bleed, x.shape[-2] - bleed, self.stride):
            for w in range(bleed, x.shape[-1] - bleed, self.stride):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)


                #print(h, h-bleed, w, w-bleed, type(out), type(patch),
                #      patch.shape, out[:, :, h-bleed, w-bleed].shape)
                out[:, :, h-bleed, w-bleed] = patch.to(out.device)

        return out

class SparseCodingLayer_First_ArchTopK(SparseCodingLayer_ArchTopK):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, padding=padding, stride=stride, k=k)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        aux = self.decoder(aux)
        aux = self.sigmoid(aux)
        aux_loss = self.mse(self.embiggen(aux_in, aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        x = self.reducer(x)

        return x, aux_loss

class SparseCodingLayer_AfterSparse_ArchTopK(SparseCodingLayer_ArchTopK):
    def __init__(self, in_dim, out_dim, filterset_size, k, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, padding=padding, stride=stride, k=k)

        self.bn = nn.BatchNorm2d(in_dim)
        self.aux_bn = nn.BatchNorm2d(in_dim*self.kernel_size*self.kernel_size)

    def forward(self, x):
        x = self.bn(x)

        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        aux = self.decoder(aux)
        aux = self.aux_bn(aux)
        aux_loss = self.mse(self.embiggen(aux_in.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        x = self.reducer(x)

        return x, aux_loss


class SparseCodingLayer_ArchOMP(nn.Module):
    def __init__(self, in_dim, out_dim, filterset_size, padding, stride):
        super().__init__()
        self.out_dim = out_dim

        self.kernel_size = 3
        self.padding = padding
        self.stride = stride

        levels, atoms_per_level = self.filterset_to_levels(filterset_size)

        self.encoder = HierarchicalRingOMP(in_dim, levels=levels, ks=[round(int(dim/k_div)) for dim in atoms_per_level])
        self.reducer = nn.Conv2d(np.sum(atoms_per_level), out_dim, kernel_size=1, stride=1, padding=0)

        self.mse = nn.MSELoss()
    
    def forward(self, x):
        raise NotImplementedError

    def filterset_to_levels(self, target):
        cur = 1
        atoms = [8]
        while np.sum(atoms) < target:
            new_atoms = atoms[-1]*2
            if np.sum(atoms)+new_atoms < target:
                cur += 1
                atoms.append(new_atoms)
                continue
            else:
                diff_below = np.abs(np.sum(atoms) - target)
                diff_above = np.abs(np.sum(atoms) + new_atoms - target)

                if diff_below <= diff_above:
                    break
                else:
                    cur += 1
                    atoms.append(new_atoms)
                    break
        return cur, atoms

    def embiggen(self, x, target):
        bleed = (self.kernel_size - 1) // 2
        out = torch.zeros_like(target)
        if self.padding > 0:
            padded = torch.zeros((x.shape[0], x.shape[1], x.shape[2]+(2*self.padding), x.shape[3]+(2*self.padding)))
            padded[:, :, self.padding:-self.padding, self.padding:-self.padding] = x
            x = padded
        for h in range(bleed, x.shape[-2] - bleed, self.stride):
            for w in range(bleed, x.shape[-1] - bleed, self.stride):
                patch = x[:, :, h-bleed:h+bleed+1, w-bleed:w+bleed+1].contiguous().view(x.shape[0], -1)


                #print(h, h-bleed, w, w-bleed, type(out), type(patch),
                #      patch.shape, out[:, :, h-bleed, w-bleed].shape)
                out[:, :, h-bleed, w-bleed] = patch.to(out.device)

        return out

class SparseCodingLayer_First_ArchOMP(SparseCodingLayer_ArchOMP):
    def __init__(self, in_dim, out_dim, filterset_size, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, padding=padding, stride=stride)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        aux = self.encoder.decode(aux)
        aux = self.sigmoid(aux)
        aux_loss = self.mse(self.embiggen(aux_in, aux).detach(), aux)
        aux = None

        x = self.encoder(x)
        x = self.reducer(x)

        return x, aux_loss

class SparseCodingLayer_AfterSparse_ArchOMP(SparseCodingLayer_ArchOMP):
    def __init__(self, in_dim, out_dim, filterset_size, padding=0, stride=1):
        super().__init__(in_dim, out_dim, filterset_size, padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(in_dim)
        self.aux_bn = nn.BatchNorm2d(in_dim*self.kernel_size*self.kernel_size)

    def forward(self, x):
        x = self.bn(x)

        aux_in = x.detach().clone()
        aux = self.encoder(aux_in)
        aux = self.encoder.decode(aux)
        aux = self.aux_bn(aux)
        aux_loss = self.mse(self.embiggen(aux_in.detach(), aux).detach(), aux)
        aux = None

        x = self.encoder(x)
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
""" Non Iterative
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
"""
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


class Conv6_SparseFirst_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 1

        out_dim = 64
        # 248 Sparse channels, akin to the 64*4=256 sparse channels of the non-hierarchical version.
        self.layer0 = sparseclassfirst(in_dim=3, out_dim=out_dim, filterset_size=round(int(out_dim*filter_set_mult)), k=k)
        self.layers = [self.layer0]
        self.non_aux = [self.layer1, self.layer2, self.layer3, self.layer4,
                        self.layer5]

class Conv6_Sparse01_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 2

        self.layer0 = sparseclassfirst(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=k)
        self.layer1 = nn.Sequential(
            sparseclass(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=k),
            CustomMaxPool(2))

        self.layers = [self.layer0, self.layer1]
        self.non_aux = [self.layer2, self.layer3, self.layer4, self.layer5]

class Conv6_Sparse012_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 3

        self.layer0 = sparseclassfirst(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=k)
        self.layer1 = nn.Sequential(
            sparseclass(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=k),
            CustomMaxPool(2))
        self.layer2 = sparseclass(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=k)


        self.layers = [self.layer0, self.layer1, self.layer2]
        self.non_aux = [self.layer3, self.layer4, self.layer5]

class Conv6_Sparse0123_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 4

        self.layer0 = sparseclassfirst(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=k)
        self.layer1 = nn.Sequential(
            sparseclass(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=k),
            CustomMaxPool(2))
        self.layer2 = sparseclass(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=k)
        self.layer3 = sparseclass(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=k)


        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]
        self.non_aux = [self.layer4, self.layer5]

class Conv6_Sparse01234_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 5

        self.layer0 = sparseclassfirst(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=k)
        self.layer1 = nn.Sequential(
            sparseclass(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=k),
            CustomMaxPool(2))
        self.layer2 = sparseclass(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=k)
        self.layer3 = sparseclass(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=k)
        self.layer4 = nn.Sequential(
            sparseclass(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=k),
            CustomMaxPool(2))


        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4]
        self.non_aux = [self.layer5]

class Conv6_Sparse012345_Hierarchical(Conv6):
    def __init__(self, filter_set_mult, k, usecase, omp):
        super().__init__(usecase)
        if omp:
            sparseclassfirst = SparseCodingLayer_First_ArchOMP
            sparseclass = SparseCodingLayer_AfterSparse_ArchOMP
        else:
            sparseclassfirst = SparseCodingLayer_First_ArchTopK
            sparseclass = SparseCodingLayer_AfterSparse_ArchTopK

        self.num_aux_losses = 6

        self.layer0 = sparseclassfirst(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=k)
        self.layer1 = nn.Sequential(
            sparseclass(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=k),
            CustomMaxPool(2))
        self.layer2 = sparseclass(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=k)
        self.layer3 = sparseclass(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=k)
        self.layer4 = nn.Sequential(
            sparseclass(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=k),
            CustomMaxPool(2))
        self.layer5 = sparseclass(in_dim=192, out_dim=256, filterset_size=round(int(256*filter_set_mult)), k=k)

        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4, self.layer5]
        self.non_aux = []



class Conv6_SparseFirst(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 1

        out_dim = 64
        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=out_dim, filterset_size=round(int(out_dim*filter_set_mult)), k=round(int(out_dim/k_div)))

        self.layers = [self.layer0]
        self.non_aux = [self.layer1, self.layer2, self.layer3, self.layer4,
                        self.layer5]

class Conv6_Sparse01(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)


        self.num_aux_losses = 2

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))

        self.layers = [self.layer0, self.layer1]
        self.non_aux = [self.layer2, self.layer3, self.layer4, self.layer5]

class Conv6_Sparse012(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 3

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))


        self.layers = [self.layer0, self.layer1, self.layer2]
        self.non_aux = [self.layer3, self.layer4, self.layer5]

class Conv6_Sparse0123(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 4

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))


        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3]
        self.non_aux = [self.layer4, self.layer5]

class Conv6_Sparse01234(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 5

        self.layer0 = SparseCodingLayer_First(in_dim=3, out_dim=64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        self.layer1 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=64, out_dim=96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))
        self.layer2 = SparseCodingLayer_AfterSparse(in_dim=96, out_dim=128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div)))
        self.layer3 = SparseCodingLayer_AfterSparse(in_dim=128, out_dim=160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div)))
        self.layer4 = nn.Sequential(
            SparseCodingLayer_AfterSparse(in_dim=160, out_dim=192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))


        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4]
        self.non_aux = [self.layer5]

class Conv6_Sparse012345(Conv6):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 6

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

        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3,
                       self.layer4, self.layer5]
        self.non_aux = []


class Conv12_Sparse0(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 2

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layers = [self.layer0_0, self.layer0_1]
        self.non_aux = [self.layer1_0, self.layer1_1,
                        self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse01(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 4

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))

        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1]

        self.non_aux = [self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse012(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 6

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1]
    
        self.non_aux = [self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse0123(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 8

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1]
    
        self.non_aux = [self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse01234(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 10

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layer4_0 = nn.Sequential(
            SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                       self.layer4_0, self.layer4_1]
    
        self.non_aux = [self.layer5_0, self.layer5_1]

class Conv12_Sparse012345(Conv12):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 12

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layer4_0 = nn.Sequential(
            SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))

        self.layer5_0 = nn.Sequential(
            SparseCodingLayer_First(192, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div))),
            CustomPad(1))
        self.layer5_1 = nn.Sequential(SparseCodingLayer_AfterSparse(256, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div))))

        
        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                       self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]
        self.non_aux = []


class Conv12_Sparse0_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 2

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layers = [self.layer0_0, self.layer0_1]
        self.non_aux = [self.layer1_0, self.layer1_1,
                        self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse01_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 4

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))

        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1]

        self.non_aux = [self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse012_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 6

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1]
    
        self.non_aux = [self.layer3_0, self.layer3_1,
                        self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse0123_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 8

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1]
    
        self.non_aux = [self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]

class Conv12_Sparse01234_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 10

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layer4_0 = nn.Sequential(
            SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))


        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                       self.layer4_0, self.layer4_1]
    
        self.non_aux = [self.layer5_0, self.layer5_1]

class Conv12_Sparse012345_Res(Conv12_Resnet):
    def __init__(self, filter_set_mult, k_div, usecase):
        super().__init__(usecase)

        self.num_aux_losses = 12

        self.layer0_0 = nn.Sequential(
            SparseCodingLayer_First(3, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))),
            CustomPad(1))
        self.layer0_1 = nn.Sequential(SparseCodingLayer_AfterSparse(64, 64, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div))))

        self.layer1_0 = nn.Sequential(
            SparseCodingLayer_First(64, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomPad(1))
        self.layer1_1 = nn.Sequential(SparseCodingLayer_AfterSparse(96, 96, filterset_size=round(int(96*filter_set_mult)), k=round(int(96/k_div))),
            CustomMaxPool(2))


        self.layer2_0 = nn.Sequential(
            SparseCodingLayer_First(96, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))),
            CustomPad(1))
        self.layer2_1 = nn.Sequential(SparseCodingLayer_AfterSparse(128, 128, filterset_size=round(int(128*filter_set_mult)), k=round(int(128/k_div))))

        self.layer3_0 = nn.Sequential(
            SparseCodingLayer_First(128, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))),
            CustomPad(1))
        self.layer3_1 = nn.Sequential(SparseCodingLayer_AfterSparse(160, 160, filterset_size=round(int(160*filter_set_mult)), k=round(int(160/k_div))))


        self.layer4_0 = nn.Sequential(
            SparseCodingLayer_First(160, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomPad(1))
        self.layer4_1 = nn.Sequential(SparseCodingLayer_AfterSparse(192, 192, filterset_size=round(int(192*filter_set_mult)), k=round(int(192/k_div))),
            CustomMaxPool(2))

        self.layer5_0 = nn.Sequential(
            SparseCodingLayer_First(192, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div))),
            CustomPad(1))
        self.layer5_1 = nn.Sequential(SparseCodingLayer_AfterSparse(256, 256, filterset_size=round(int(256*filter_set_mult)), k=round(int(256/k_div))))

        
        self.layers = [self.layer0_0, self.layer0_1, self.layer1_0, self.layer1_1,
                       self.layer2_0, self.layer2_1, self.layer3_0, self.layer3_1,
                       self.layer4_0, self.layer4_1, self.layer5_0, self.layer5_1]
        self.non_aux = []





""" Non Iterative
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

        aux_losses = torch.mean(torch.stack((aux_loss_00, aux_loss_01,
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
"""



""" Non Iterative
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
"""


class Conv6_NonIterative(nn.Module):
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

        logits, preds = self.classify(x)
        return logits, preds, None



class Conv6_Sparse01_NonIterative(Conv6_NonIterative):
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

class Conv6_Sparse012_NonIterative(Conv6_NonIterative):
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

class Conv6_Sparse0123_NonIterative(Conv6_NonIterative):
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

class Conv6_Sparse01234_NonIterative(Conv6_NonIterative):
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
       
        #ls =[aux_loss_0, aux_loss_1, aux_loss_2, aux_loss_3, aux_loss_4]

        #for loss in ls:
        #    print(loss.item())
        #print(torch.stack(tuple(ls)).shape)

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

class Conv6_Sparse012345_NonIterative(Conv6_NonIterative):
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

        aux_losses = torch.stack((aux_loss_5,aux_loss_4, aux_loss_3, aux_loss_2, aux_loss_1, aux_loss_0))
        aux_loss = torch.mean(aux_losses)
        #aux_loss = torch.mean(aux_losses.detach())

        if self.usecase == "pretrain" or self.usecase == "random":
            x = x.detach().clone()
            x.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None
        #REMOVEDBATCHNORM
        #features = self.classify_bnorm(x) 
        features = x.view(x.shape[0], -1)
        #features = features.view(features.shape[0], -1)
        logits = self.classify_fc(features)

        _, preds = logits.max(dim=1)
        return logits, preds, aux_loss
