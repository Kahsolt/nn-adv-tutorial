#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/21

from typing import *
from PIL import Image
import PIL.Image as PILImage

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad
import torchvision.transforms.functional as TF
from torchvision.models.resnet import resnet18
import numpy as np
import torchattacks

npimg = np.ndarray

FILE = 'ILSVRC2012_val_00000031.png'


def load_img(fp) -> PILImage:
  # to RGB
  return Image.open(fp).convert('RGB')

def pil_to_im(img:PILImage) -> npimg:
  # dtype convert: uint8 -> float32
  return np.asarray(img, dtype=np.float32) / 255.

def im_to_tensor(im:npimg) -> Tensor:
  # shape convert: [H, W, C] => [C, H, W]
  return torch.from_numpy(im).permute([2, 0, 1])


def imagenet_stats() -> Tuple[Tuple[float], Tuple[float]]:
  mean = (0.485, 0.456, 0.406)
  std  = (0.229, 0.224, 0.225)
  return mean, std

def normalize(X: Tensor) -> Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until enumerating a dataloader '''
  mean, std = imagenet_stats()
  return TF.normalize(X, mean, std)       # [B, C, H, W]


def fgsm(model, X:Tensor, Y:Tensor, eps:float=8/255):
  # X.shape: [B=1, C=3, H=224, W224]
  X.requires_grad = True
  logits = model(normalize(X))   # [B=1, NC=1000]
  loss = F.cross_entropy(logits, Y)
  g = grad(loss, X, loss)[0]   # Jaccob's [B=1, C, H, W]
  AX = X + g.sign() * eps
  return AX


def pgd(model, X:Tensor, Y:Tensor, eps:float=8/255, alpha=1/255, steps:int=20):
  X_orig = X.detach().clone()
  AX = X_orig.detach().clone() + (torch.rand_like(X_orig) * 2 - 1) * eps
  for i in range(steps):
    AX.requires_grad = True
    logits = model(normalize(AX))
    pred = logits.argmax(dim=-1)
    print('>> pred:', pred)
    loss = F.cross_entropy(logits, Y, reduction='none')   # [B]
    g = grad(loss, AX, loss)[0]
    AX_new = AX.detach() + g.sign() * alpha
    DX = (AX_new - X_orig).clamp(-eps, eps)
    AX = (X_orig + DX).clamp(0, 1).detach()
  return AX


model = resnet18(pretrained=True)
#X = torch.randn([1, 3, 224, 224])
img = load_img(FILE)
im = pil_to_im(img)
X = im_to_tensor(im).unsqueeze(dim=0)    # [B=1]

logits = model(X)
pred = logits.argmax(dim=-1)
print('X:', pred)
Y = pred

AX = fgsm(model, X, Y)
DX = AX - X
logits_AX = model(AX)
pred_AX = logits_AX.argmax(dim=-1)
print('AX:', pred_AX)

AX = pgd(model, X, Y)
logits_AX = model(AX)
pred_AX = logits_AX.argmax(dim=-1)
print('AX:', pred_AX)

Linf = (AX - X).abs().max()
L1 = (AX - X).abs().mean()
L2 = ((AX.flatten() - X.flatten())**2).sum().sqrt()


# finite diff:
#   lim[h->0] (f(x+h) - f(x)) / h
# parameter shift:
#   f'(x)
fx = model(X)
h = torch.zeros_like(X)
h[0, 0, 0, 0] += 0.01
fx_h = model(X + h)

g = (fx - fx_h) / 0.01
