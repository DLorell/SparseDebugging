import torch
import torch.nn as nn
import numpy as np
import src.functional as f
import src.nonlinearities



class HierarchicalRingTopK(nn.Module):
    def __init__(self, in_ch, levels, ks):
        super().__init__()
        connectivity = 4

        self.atoms_per_level = [8]
        for _ in range(levels-1):
            self.atoms_per_level.append(self.atoms_per_level[-1]*2)
        total_atoms = np.sum(self.atoms_per_level)
        self.connections = self.form_connections(self.atoms_per_level, connectivity)

        self.dict = nn.Conv2d(in_channels=in_ch, out_channels=total_atoms, kernel_size=3, stride=1, padding=0, bias=True)
        with torch.no_grad():
            f.orthonormalize_init(self.dict)

        self.selector = self.topk
        self.ks = ks

    def forward(self, x):
        all_activations = self.dict(x)
        descending_mask = None
        masked_activations = None
        for i, (atomic_activations, atoms) in enumerate(zip(self.activation_splitter(all_activations), self.atomic_splitter())):
            k = self.ks[i]
            if descending_mask is not None: atomic_activations = self.gate(i-1, atomic_activations, descending_mask)
            selected_activations = self.selector(atomic_activations, atoms.view(atoms.shape[0], -1).T, k)
            descending_mask = (selected_activations != 0).float()
            if masked_activations is None:
                masked_activations = selected_activations
            else:
                masked_activations = torch.cat([masked_activations, selected_activations], dim=1) 
        return masked_activations

    def gate(self, level, activations, descending_mask):
        connections = self.connections[level]
        gating_tensor = torch.zeros_like(activations)
        for i, idxs in enumerate(connections):
            mask = descending_mask[:, i, :, :]
            for idx in idxs:
                gating_tensor[:, idx, :, :] += mask
        or_gate = (gating_tensor > 0).float()
        return activations * or_gate

    def activation_splitter(self, activations):
        idx = 0
        for num_atoms in self.atoms_per_level:
            atoms = activations[:, idx:idx+num_atoms, : ,:]
            idx = idx+num_atoms
            yield atoms
    
    def atomic_splitter(self):
        idx = 0
        for num_atoms in self.atoms_per_level:
            atoms = self.dict.weight[idx:idx+num_atoms, :, : ,:]
            idx = idx+num_atoms
            yield atoms

    def form_connections(self, atoms_per_level, connectivity):
        connections_per_level = []
        for i, atoms in enumerate(atoms_per_level[:-1]):
            atoms_in_next_level = atoms_per_level[i+1]
            connections_per_level.append([])
            for j in range(atoms):
                connections = [k % atoms_in_next_level for k in range(j*2, connectivity+j*2)]
                connections_per_level[i].append(connections)
        connections_per_level.append(None)
        return connections_per_level
    
    def topk(self, x, _, k):
        top_vals, top_idxs = torch.topk(torch.abs(x), k, dim=1, sorted=False)
        out = torch.zeros_like(x) + -999.99
        out.scatter_(dim=1, index=top_idxs, src=top_vals)
        mask = (out != -999.99)
        out = x * mask
        return out

class HierarchicalRingOMP(HierarchicalRingTopK):
    def __init__(self, in_ch, levels, ks):
        super().__init__(in_ch, levels=levels, ks=ks)
        self.selector = src.nonlinearities.BatchOMP()

        # Create "decoder" using the same parameters as the encoding dict.

        dict_shape = self.dict.weight.shape
        sparse_size = dict_shape[0]
        patch_size = dict_shape[1] * dict_shape[2] * dict_shape[3]
        self.decoder = nn.Conv2d(in_channels=sparse_size, out_channels=patch_size, kernel_size=1, bias=False)
        self.decoder.weight = nn.parameter.Parameter(
            self.dict.weight.view(sparse_size, -1, 1, 1).permute(1,0,2,3)
        )

    def decode(self, descriptor):
        non_zeros = (descriptor != 0).float()
        debiased = (descriptor - (self.dict.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))) * non_zeros
        recon = self.decoder(descriptor)
        return recon




