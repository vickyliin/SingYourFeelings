#!/usr/bin/env python
import dataset
from config import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import word2vec

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

class MusicEncoder(nn.Module):
  L, Ci, E, K, Co, M = music.L, music.Ci, music.E, music.K, music.Co, M
  dp = music.dp
  assert L >= K, 'MusicEncoder: Seq length should >= kernel size'
  @param(
    note = ['Tensor', (300, Ci, L, E)], 
    tempo = ['Tensor', (300, Ci)],
  )
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(self.Ci, self.Co, (self.K, self.E))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(self.Co, self.M-1)
    self.dropout = nn.Dropout(self.dp, inplace=True)

  def forward(self, inp):
    note, tempo = inp
    # note: torch tensor, N x Ci x L x E
    # tempo: torch tensor, N x Ci
    # out: torch tensor variable, N x M
    assert type(note).__name__.endswith('Tensor')
    note = self.note.resize_(note.size()).copy_(note)
    note = Variable(note)
    assert type(tempo).__name__.endswith('Tensor')
    tempo = self.tempo.resize_(tempo.size()).copy_(tempo)
    tempo = Variable(tempo)

    hid = self.conv(note).squeeze(-1)             # N x Co x L-
    hid = self.pool(hid).squeeze(-1)              # N x Co
    hid = self.linear(hid)                        # N x M-1
    self.dropout(hid)

    out = torch.cat([tempo.sum(1), hid], 1)       # N x M

    return out

class MusicDecoder(nn.Module):
  L, Ci, E, K, Co, M = music.L, music.Ci, music.E, music.K, music.Co, M
  dp = music.dp
  assert L >= K, 'MusicDecoder: Seq length should >= kernel size'
  @param()
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(self.M, self.Co)
    self.unpool = nn.Linear(1, self.L+self.K-1)
    self.unconv = nn.Conv1d(self.Co, self.Ci*self.E, self.K)
    self.dropout = nn.Dropout(self.dp, inplace=True)

  def forward(self, inp):
    # inp: torch tensor variable, N x M
    # out: torch tensor variable, N x Ci x L x E
    assert type(inp).__name__ == 'Variable'

    hid = self.linear(inp)                        # N x Co
    hid = self.unpool(hid.view(-1,1))             # N*Co x L+K-1
    hid = hid.view(-1, self.Co, self.L+self.K-1)  # N x Co x L+K-1
    self.dropout(hid)

    out = self.unconv(hid)                        # N x Ci*E x L
    out = out.view(-1, self.Ci, self.E, self.L)   # N x Ci x E x L
    out = out.transpose(2, 3).contiguous()        # N x Ci x L x E
    self.dropout(out)

    return out


class LyricsEncoder(nn.Module):
  L, E, K, Co, M = lyrics.L, lyrics.E, lyrics.K, lyrics.Co, M
  dp = lyrics.dp
  assert L >= K, 'LyricsEncoder: Seq length should >= kernel size'
  @param(inp = ['LongTensor', (300, L)])
  def __init__(self, vs):
    # vs: vocabulary size
    super().__init__()
    self.embed = nn.Embedding(vs, self.E)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    self.conv = nn.Conv2d(1, self.Co, (self.K, self.E))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(self.Co, self.M)


  def forward(self, inp):
    # inp: torch tensor, N x L
    # out: torch tensor variable, N x M
    assert type(inp).__name__.endswith('Tensor')
    inp = self.inp.resize_(inp.size()).copy_(inp)
    inp = Variable(inp)

    emb = self.embed(inp).unsqueeze(1)            # N x 1 x L x E
    self.dropout(emb)

    hid = self.conv(emb).squeeze(-1)              # N x Co x L-
    hid = self.pool(hid).squeeze(-1)              # N x Co
    self.dropout(hid)

    out = self.linear(hid)                        # N x M
    self.dropout(out)

    return out

def translateWrap(translator):
  def translate(s):
    inp = dataset.lyr2vec(s)
    dec = translator(inp)
    out = datast.vec2midi(dec)
    return out

  return translate

class Translator(nn.Module):
  @param()
  def __init__(self, init = None):
    # encoder: lyrics encoder / music encoder
    # decoder: music decoder
    super().__init__()
    if init:
      encoder, decoder = init
      self.encoder = encoder
      self.decoder = decoder
    else:
      vs = len(dataset.lex.vocab)
      self.encoder = LyricsEncoder(vs)
      self.decoder = MusicDecoder()
      self.translate = translateWrap(self)
      self.train(False)

  def forward(self, inp):
    enc = self.encoder(inp)
    dec = self.decoder(enc)
    return dec


if __name__ == '__main__':
  vsL, vsM = len(dataset.lex.vocab), 5
  n = 3
  lyr = torch.floor( torch.rand(n, lyrics.L) * vsL )
  note = torch.floor( torch.rand(n, music.Ci, music.L, music.E) * vsM )
  tempo = torch.floor( torch.rand(n, music.Ci) * vsM )
  mus = (note, tempo)

  le = LyricsEncoder(vsL)
  print(le)
  outLe = le(lyr)
  assert outLe.size(0) == n
  assert outLe.size(1) == M

  me = MusicEncoder()
  print(me)
  outMe = me(mus)
  assert outMe.size(0) == n
  assert outMe.size(1) == M

  md = MusicDecoder()
  print(md)
  outMd = md(outMe)
  assert outMd.size() == note.size()

  ae = Translator([me, md])
  print('Autoencoder: ', ae)
  outAe = ae(mus)
  assert outAe.size() == note.size()

  tr = Translator([le, md])
  print('Translator: ', tr)
  outTr = tr(lyr)
  assert outTr.size() == note.size()

  torch.save(tr.state_dict(), 'model/test.para')
