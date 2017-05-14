#!/usr/bin/env python

import torch
from torch.autograd import Variable
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class param:
  def __init__(self, **args):
    self.args = args

  def __call__(self, fin):
    params = self.args
    def __init__(self, *args, **kwargs):
      fin(self, *args, **kwargs)
      if use_cuda:
        self.cuda()
        for k, (t, v) in params.items():
          print(k, ':', t, v)
          setattr(self, k, getattr(torch,t)(*v).cuda())
      else:
        for k, v in params.items():
          print(k, ':', t, v)
          setattr(self, k, getattr(torch,t)(*v))

    return __init__

def load(filename):
  if use_cuda:
    sd = torch.load(filename)
  else:
    sd = torch.load(filename, 
      map_location = lambda storage, loc: storage)

  return sd

# max music length
L = 300
# music channel number
Ci = 5
# embedding size
E = 50
# CNN kernel size
K = 10
# CNN output channel
Co = 200

# encoded vector size
M = 100


class MusicEncoder(nn.Module):
  L, Ci, E, K, Co, M = L, Ci, E, K, Co, M
  # Dropout rate
  dp = .5
  @param(inp = ['LongTensor', (300, Ci, L)])
  def __init__(self, vs):
    # vs: vocabulary size
    super().__init__()
    self.embed = nn.Embedding(vs, self.E)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    self.conv = nn.Conv2d(self.Ci, self.Co, (self.K, self.E))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(self.Co, self.M)


  def forward(self, inp):
    # inp: torch tensor, N x Ci x L
    # out: torch tensor, N x M
    assert type(inp).__name__.endswith('Tensor')
    inp = self.inp.resize_(inp.size()).copy_(inp)
    inp = Variable(inp)

    emb = self.embed(inp.view(-1, self.L))        # N*Ci x L x E
    emb = emb.view(-1, self.Ci, self.L, self.E)   # N x Ci x L x E
    self.dropout(emb)

    hid = self.conv(emb).squeeze(-1)              # N x Co x L-
    hid = self.pool(hid).squeeze(-1)              # N x Co
    self.dropout(hid)

    out = self.linear(hid)                        # N x M
    self.dropout(out)

    return out

class MusicDecoder(nn.Module):
  L, Ci, E, K, Co, M = L, Ci, E, K, Co, M
  # Dropout rate
  dp = .5

  @param(inp = ['FloatTensor', (300, M)])
  def __init__(self, vs):
    # vs: vocabulary size
    super().__init__()
    self.linear = nn.Linear(self.M, self.Co)
    self.unpool = nn.Linear(1, self.L)
    self.unconv = nn.Conv1d(self.Co, self.Ci*self.E, self.K)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    self.pointer = nn.Linear(self.E, vs)


  def forward(self, inp):
    # inp: torch tensor, N x M
    # out: torch tensor, N*Ci*L x V
    assert type(inp).__name__.endswith('Tensor')
    inp = self.inp.resize_(inp.size()).copy_(inp)
    inp = Variable(inp)

    hid = self.linear(inp)                        # N x Co
    hid = self.unpool(hid.view(-1,1))             # N*Co x L
    hid = hid.view(-1, self.Co, self.L)           # N x Co x L
    self.dropout(hid)

    emb = self.unconv(hid)                        # N x Ci*E x L
    emb = emb.view(-1, self.Ci, self.E, self.L)   # N x Ci x E x L
    emb = emb.transpose(2, 3).contiguous()        # N x Ci x L x E
    emb = emb.view(-1, self.E)                    # N*Ci*L x E
    self.dropout(emb)

    out = self.pointer(emb)                       # N*Ci*L x V
    self.dropout(out)

    return out


class Translator(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, inp):
    pass

  def translate(self, inp):
    pass

if __name__ == '__main__':
  me = MusicEncoder(10)
  print(me)
  print(me(me.inp))

  md = MusicDecoder(10)
  print(md)
  print(md(md.inp))
