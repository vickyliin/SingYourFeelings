#!/usr/bin/env python
import model
import dataset

import torch
import config
from copy import deepcopy

def batchLoss(model, dataset, criterion, train = True):
  Ci = model.Ci
  epoch_loss = [0, 0]
  loss = [0, 0]
  for batch in dataset:
    inp, tar = batch
    tar = model.wrapTar(tar)
    out = model(inp)

    loss[0] = criterion(out[0], tar[0])
    loss[1] = criterion(out[1], tar[1])
    if train:
      yield loss, False

    epoch_loss[0] += loss[0].data[0]
    epoch_loss[1] += loss[1].data[0]

  loss = [ (loss/dataset.size())**.5 for loss in epoch_loss ]
  loss = (loss[0]*loss[1])**.5
  fmt = '['+', '.join(['{:.4f}']*Ci)+']'
  '''
  if type(inp) == tuple:
    print('  InpTempo:', fmt.format(*inp[1][0]))
    print('  InpNotes:', fmt.format(*inp[0][0,:,0,0]))
  else:
    print('  InpLyr:', fmt.format(*inp[0]))
  print('  OutTempo:', fmt.format(*out[1].data[0]))
  print('  OutNotes:', fmt.format(*out[0].data[0,:,0,0]))
  '''
  print(' - %s: ' % ['Validate', 'Train'][train], end='')
  print('{:.4f}'.format(loss), end='')
  '''
  if not train:
    print()
  '''
  yield loss, True


def earlyStop(fin):
  def train(model, trainset, valset, args):
    printfmt = lambda i,endure: print(
      '\rEpoch {:3} - Endure {:3}/{:3}'.format(i, endure, args.endure),
      end = '')

    trainer = fin(model, trainset, valset, args)
    printfmt(1, 0)
    endure, min_loss = 0, next(trainer)
    sd = deepcopy(model.state_dict())
    printfmt(2, endure)
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
  criterion = getattr(torch.nn, args.loss)()

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

    model.train(False)
    loss, _ = next(batchLoss(model, valset, criterion, train=False))
    yield loss


if __name__ == '__main__':
  import random
  vs = len(dataset.lex.vocab)
  le = model.LyricsEncoder(vs)
  me = model.MusicEncoder()
  md = model.MusicDecoder()
  
  ae = model.Translator([me, md])
  tr = model.Translator([le, md])
  args = config.train_ae
  class Dataset(list):
    def shuffle(self):
      random.shuffle(self)
    def size(self):
      return sum([ len(batch) for batch in self ])

  AEtrainset = Dataset()
  AEvalset = Dataset()

  TRtrainset = Dataset()
  TRvalset = Dataset()

  # n: number of batch in an dataset
  n = 20
  for i in range(2*n):
    nsize = (args.batch_size, *ae.note.size()[1:])
    tsize = (args.batch_size, *ae.tempo.size()[1:])
    lsize = (args.batch_size, *tr.encoder.inp.size()[1:])

    inp = torch.rand(*lsize)

    note = torch.rand(*nsize)
    tempo = torch.rand(*tsize)
    tar = (note, tempo)

    if i < n:
      AEvalset.append(dataset.Batch([tar, tar]))
      TRvalset.append(dataset.Batch([inp, tar]))
    else:
      AEtrainset.append(dataset.Batch([tar, tar]))
      TRtrainset.append(dataset.Batch([inp, tar]))


  print(args)
  print(ae)
  train(ae, AEtrainset, AEvalset, args = args)


  print(args)
  print(tr)
  train(tr, TRtrainset, TRvalset, args = args)
