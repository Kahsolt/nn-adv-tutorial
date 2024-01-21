#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/21 

import json
from argparse import ArgumentParser
import torchattacks

import matplotlib.pyplot as plt

BASE_PATH = ''


def run_one(args):
  # X, Y = dataset[args.i]
  # pred_X = model(X)
  # AX = attack(X)
  # pred_AX = model(AX)
  #
  # DX = AX - X
  # Linf, L1, L2 = ...
  # plt.savefig([X, AX, minmax_norm(DX)])
  #
  # json.save({
  #   cmd: ' '.join(sys.args),
  #   args: vars(args),
  #   dt: ...   // run start date-time (str)
  #   ts: ...   // run time in seconds (float)
  #   pred: {
  #     truth: Y
  #     raw: pred_X,
  #     adv: pred_AX,
  #   },
  #   metric: {
  #     Linf: ...
  #     L1: ...
  #     L2: ...
  #   },
  # })
  pass


def run_all(args):
  # acc, racc, asr, psr, sasr = ...
  #
  # for X, Y in dataloader:
  #   AX = attack(X, Y)
  #   ...
  #
  # json.save({
  #   cmd: ' '.join(sys.args),
  #   args: vars(args),
  #   dt: ...   // run start date-time (str)
  #   ts: ...   // run time in seconds (float)
  #   metric: {
  #     acc: ...
  #     racc: ...
  #     asr: ...
  #     psr: ...
  #     sasr: ...
  #   },
  # })
  pass



if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-I', '--i', type=int, help='image index of the dataset')
  parser.add_argument('-M', '--method', default='PGD', choices=['FGSM', 'PGD'])
  parser.add_argument('-B', '--batch_size', default=16, type=int)
  parser.add_argument('--steps', default=10,    type=int)
  parser.add_argument('--alpha', default=1/255, type=eval)
  parser.add_argument('--eps',   default=8/255, type=eval)
  args = parser.parse_args()

  if args.i:
    run_one(args)
  else:
    run_all(args)
