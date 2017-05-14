#!/usr/bin/env python
import model
import dataset

import torch
from torch import optim, nn

class args:
  optim = optim.RMSprop
  optim_args = dict(
    lr = 3e-3
  )
  batch_size = 1
  max_epoch = 1000
  endure = 30
  loss = nn.MSELoss()


def batchLoss(model, dataset, train = True):
  epoch_loss = [0, 0]
  loss = [0, 0]
  for batch in dataset:
    inp, tar = batch
    tar = model.wrapTar(tar)
    out = model(inp)

    loss[0] = args.loss(out[0], tar[0])
    loss[1] = args.loss(out[1], tar[1])
    if train:
      yield loss, False

    epoch_loss[0] += loss[0].data[0]
    epoch_loss[1] += loss[1].data[0]

  note, tempo = inp
  print('- InpTempo\t-', tempo.squeeze().tolist())
  note, tempo = out
  print('- OutTempo\t-', tempo.data.squeeze().tolist())

  loss = [ loss/dataset.size() for loss in epoch_loss ]
  yield loss, True



def train(model, trainset, valset):
  optim = args.optim(model.parameters(), **args.optim_args)
  optim.zero_grad()

  for epoch in range(1, args.max_epoch):
    print('\nEpoch', epoch)
    trainset.shuffle()
    model.train(True)
    for loss, end in batchLoss(model, trainset):
      if end:
        break
      loss[0].backward(retain_variables=True)
      loss[1].backward()
      optim.step()
      optim.zero_grad()
    print('Train Loss\t-',
        'note: {l[0]:8.2f}, tempo {l[1]:8.2f}'.format(l=loss))

    model.train(False)
    loss, _ = next(batchLoss(model, valset, train=False))
    print('Validate Loss\t-',
        'note: {l[0]:8.2f}, tempo {l[1]:8.2f}'.format(l=loss))


if __name__ == '__main__':
  import random
  vs = len(dataset.lex.vocab)
  le = model.LyricsEncoder(vs)
  me = model.MusicEncoder()
  md = model.MusicDecoder()
  
  ae = model.Translator([me, md])
  class Dataset(list):
    def shuffle(self):
      random.shuffle(self)
    def size(self):
      return sum([ len(batch) for batch in self ])

  trainset = Dataset()
  valset = Dataset()

  # n: number of batch in an epoch
  n, max_val = 2, 100
  for i in range(n):
    nsize = (args.batch_size, *ae.note.size()[1:])
    tsize = (args.batch_size, *ae.tempo.size()[1:])

    note = torch.floor( torch.rand(*nsize) * max_val )
    tempo = torch.floor( torch.rand(*tsize) * max_val )
    inp = (note, tempo)

    note = torch.floor( torch.rand(*nsize) * max_val )
    tempo = torch.floor( torch.rand(*tsize) * max_val )
    tar = (note, tempo)

    if i < n//2:
      valset.append(dataset.Batch([inp, tar]))
    else:
      trainset.append(dataset.Batch([inp, tar]))


  train(ae, trainset, valset)

  #tr = Translator(le, md)
