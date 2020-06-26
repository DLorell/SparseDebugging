import pytest
import pytest_check as check
import sys
import torch
import torch.nn as nn
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('src')
import functional as f  # nopep8


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def save_torch_image(x, filename):
    x = x.permute(1, 2, 0).detach().cpu().numpy()
    x = x.astype('uint8')
    img = Image.fromarray(x)
    img.save("test/imgs/" + filename)


def test_batch_omp_nograd():
    with torch.no_grad():
        _, testloader = f.get_dataloaders(augment=False, batch_size=128)
        for batch in testloader:
            inputs, _ = batch
            inputs = inputs[:1].to(DEVICE)
            break

        conv = nn.Conv2d(3, 32, kernel_size=3, bias=False).to(DEVICE)
        f.orthonormalize_init(conv)
        D = conv.weight.view(conv.weight.shape[0], -1).T
        k = 2

        activations = conv(inputs)
        activations = f.batch_omp(activations, D, k)

        avg_sparsity = torch.mean(torch.sum((activations != 0).float(), dim=1))
        check.is_true(torch.abs(avg_sparsity - k) <= 4)

        activations, _ = f._batch_vectorize(activations)

        d_stack = torch.stack([D for i in range(activations.shape[0])], dim=0)
        recon_cols = torch.bmm(d_stack, activations.unsqueeze(-1)).contiguous().squeeze().view(-1, 30, 30, 27).detach().cpu().numpy()

        #recon_cols = torch.bmm(torch.stack([D for i in range(activations.T.shape[1])]), activations.T).contiguous().view(-1, 30, 30, 27).detach().cpu().numpy()

        recon_piecewise = build_reconstruction(recon_cols, kernel_size=3, ch_depth=3, avg=False)
        recon_avg = build_reconstruction(recon_cols, kernel_size=3, ch_depth=3, avg=True)

        catted_input = normalize_image(torch.cat(list(inputs), dim=1).float().cpu())
        cattted_recon_piecewise = normalize_image(torch.tensor(recon_piecewise).permute(2,0,1).float().cpu())
        cattted_recon_avg = normalize_image(torch.tensor(recon_avg).permute(2,0,1).float().cpu())
        final = torch.cat([cattted_recon_avg, catted_input, cattted_recon_piecewise], dim=-1)

        save_torch_image(final, "out.png")


def build_reconstruction(x, kernel_size, ch_depth=3, avg=False):

    img_h = x.shape[1]
    img_w = x.shape[2]

    #imgs = [x]
    imgs = [img for img in x]

    bleed = (kernel_size - 1) // 2

    final_h = imgs[0].shape[1] + 2*bleed
    final_w = img_w + 2*bleed

    avg_imgs = []
    for recon in imgs:
        avg_img = np.zeros((final_h,final_w,ch_depth))
        avg_divisors = np.zeros((final_h,final_w,ch_depth))
        for h in range(0, recon.shape[0], 1):
            for w in range(0, recon.shape[1], 1):
                patch = torch.tensor(recon[h, w]).view(ch_depth, kernel_size, kernel_size).permute(1, 2 ,0).cpu().numpy()
                ones = np.ones_like(patch)
                center_h = h + bleed
                center_w = w + bleed

                if avg:
                    avg_img[center_h-bleed:center_h+bleed+1, center_w-bleed:center_w+bleed+1, :] += patch
                else:
                    avg_img[center_h-bleed:center_h+bleed+1, center_w-bleed:center_w+bleed+1, :] = patch

                avg_divisors[center_h-bleed:center_h+bleed+1, center_w-bleed:center_w+bleed+1, :] += ones
        if avg:
            avg_img = avg_img / avg_divisors
        avg_imgs.append(avg_img)

    avg_img = np.concatenate(avg_imgs, axis=0)

    return avg_img

def normalize_image(x):
    x = x - torch.min(x)
    x = x / torch.max(x)
    x = x * 255
    return x
