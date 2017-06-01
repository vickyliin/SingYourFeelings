#!/usr/bin/env python
import torch
from copy import deepcopy

import config
import model
import dataset

def batchLoss(model, dataset, criterion, train=True):
  epoch_loss = [0, 0]
  loss = [0, 0]
  for batch in dataset:
    inp, tar = batch
    tar = model.wrapTar(tar)
    out = model(inp)

    loss[0] = criterion[0](out[0].view(-1,config.note.size), tar[0].view(-1))
    loss[1] = criterion[1](out[1], tar[1])
    if train:
      yield loss, False

    epoch_loss[0] += loss[0].data[0]
    epoch_loss[1] += loss[1].data[0]

  loss = [ (loss/len(dataset))**.5 for loss in epoch_loss ]

  _loss = (loss[0]*loss[1])**.5

  print(' - %s: ' % ['Validate', 'Train'][train], end='')
  print('{:.4f} {:.4f} : {:.4f}'.format(loss[0], loss[1], _loss), end='')
  yield _loss, True

def validate(model, valset, criterion):
  return next(batchLoss(model, valset, criterion, train=False))[0]


def earlyStop(fin):
  def train(model, trainset, valset, args):
    fmt = '\rEpoch {:3} - Endure {:3}/{:3}'
    def printfmt(i, endure): print(fmt.format(i, endure, args.endure), end='')

    trainer = fin(model, trainset, valset, args)
    printfmt(1, 0)
    endure, min_loss = 0, float('inf')
    for i, loss in enumerate(trainer, 3):
      printfmt(i, endure)
      if loss < min_loss:
        min_loss = loss
        endure = 0
        sd = deepcopy(model.state_dict())
      else:
        endure += 1
        if endure > args.endure:
          model.load_state_dict(sd)
          print('\nmin Validate Loss: {:.4f}'.format(min_loss))
          break

    
  return train


@earlyStop
def train(model, trainset, valset, args):
  optim = getattr(torch.optim, args.optim)
  optim = optim(model.parameters(), **args.optim_args)
  optim.zero_grad()
  criterion = (getattr(torch.nn, args.loss_cate)(), getattr(torch.nn, args.loss_val)())

  for epoch in range(1, args.max_epoch+1):
    trainset.shuffle()
    model.train(True)
    for loss, end in batchLoss(model, trainset, criterion):
      if end:
        break
      loss[0].backward(retain_variables=True)
      loss[1].backward()
      optim.step()
      optim.zero_grad()
      '''
      print('Loss: {:.4f}, {:.4f}'.format(
          loss[0].data[0], loss[1].data[0]), end='\r')
      '''

    model.train(False)
    valloss = validate(model, valset, criterion)
    yield valloss


if __name__ == '__main__':
  AEtrainset, TRtrainset = dataset.load('data/train.jsonl')
  AEvalset, TRvalset = dataset.load('data/valid.jsonl')

  vs = dataset.vs
  le = model.LyricsEncoder(vs)
  me = model.MusicEmbEncoder()
  md = model.MusicEmbDecoder()
  
  ae = model.Translator([me, md])
  tr = model.Translator([le, md])

  args = config.autoencoder
  print(args)
  print(ae)
  train(ae, AEtrainset, AEvalset, args = args)


  args = config.translator
  print(args)
  print(tr)
  train(tr, TRtrainset, TRvalset, args = args)

  filename = 'model/%s.para'%args.name
  torch.save(tr.state_dict(), 'model/%s.para'%args.name)
  
  sd = model.load(filename)
  for name, para in tr.named_parameters():
    assert all(sd[name].view(-1) == para.data.view(-1))
