#!/usr/bin/env python
import config

import torch
from torch.autograd import Variable
import torch.nn as nn
import word2vec as w2v

#use_cuda = False
use_cuda = torch.cuda.is_available()

lex = w2v.load(config.lyrics.lex)
vs = len(lex.vocab)

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
          setattr(self, k, getattr(torch,t)(*v).cuda())
      else:
        for k, (t, v) in params.items():
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
  @param(
    note = ['LongTensor', (300, config.music.L)], 
    tempo = ['Tensor', (300,)],
  )
  def __init__(self):
    super().__init__()
    self.L, self.M = config.music.L, config.M
    self.Emb, self.Siz = config.note.dim, config.note.size
    self.dp = config.music.dp
    # K, Co = config.music.K, config.music.Co
    # assert L >= K, 'MusicEncoder: Seq length should >= kernel size'

    self.emb = nn.Embedding(self.Siz, self.Emb)
    '''
    self.conv = nn.Conv2d(1, self.Co, (self.K, self.Emb))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(1+self.Co, self.M)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    '''
    self.activate = nn.Sigmoid()
    self.QAQ = nn.Linear(1+self.L*self.Emb, self.M)

  def forward(self, inp):
    note, tempo = inp
    # note: torch tensor, N x L
    # tempo: torch tensor, N
    # out: torch tensor variable, N x M
    assert type(note).__name__.endswith('Tensor')
    note = self.note.resize_(note.size()).copy_(note)
    note = Variable(note)
    assert type(tempo).__name__.endswith('Tensor')
    tempo = self.tempo.resize_(tempo.size()).copy_(tempo)
    tempo = Variable(tempo)

    note = self.emb(note).view(-1, self.L*self.Emb)
    hid = torch.cat([tempo.view(-1,1), note], 1)
    out = self.QAQ(hid)
    out = self.activate(out)

    return out

    '''
    note = self.emb(note.view(-1,self.Ci*self.L)).view(-1, self.self.L, self.Emb) # N x Ci x L x Emb
    hid = self.conv(note).squeeze(-1)             # N x Co x L-
    hid = self.pool(hid).squeeze(-1)              # N x Co
    self.dropout(hid)

    hid = torch.cat([tempo.view(-1,1), hid], 1)   # N x 1+Co

    out = self.linear(hid)                        # N x M
    self.dropout(out)
    self.activate(out)

    return out
    '''

class MusicDecoder(nn.Module):
  @param()
  def __init__(self):
    super().__init__()
    self.L, self.K, self.Co, self.M = config.music.L, config.music.K, config.music.Co, config.M
    self.Emb, self.Siz = config.note.dim, config.note.size
    self.dp = config.music.dp
    assert self.L >= self.K, 'MusicDecoder: Seq length should >= kernel size'

    self.linear = nn.Linear(self.M, 1+self.Co)
    self.unpool = nn.Linear(1, self.L+self.K-1)
    self.unconv = nn.Conv1d(self.Co, self.Emb, self.K)
    self.unemb = nn.Linear(self.Emb, self.Siz)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    self.activate = nn.ReLU()
    '''
    self.soft = nn.Softmax()
    self.QAQ = nn.Linear(self.M, 1+self.L*self.Siz)
    '''

  def forward(self, inp):
    # inp: torch tensor variable, N x M
    # note: torch tensor variable, N*L x Siz
    # tempo: torch tensor variable, N
    assert type(inp).__name__ == 'Variable'

    hid = self.linear(inp)                        # N x 1+Co
    tempo, hid = hid[:,0], hid[:,1:]              # N x Co
    hid = hid.contiguous()
    hid = self.unpool(hid.view(-1,1))             # N*Co x L+K-1
    hid = hid.view(-1, self.Co, self.L+self.K-1)  # N x Co x L+K-1
    hid = self.activate(hid)

    note = self.unconv(hid)                       # N x Emb x L
    self.dropout(note)
    note = note.transpose(1, 2).contiguous()      # N x L x Emb
    note = self.unemb(note.view(-1,self.Emb))     # N*L x Siz
    self.dropout(note)

    # tempo = self.activate(tempo)
    # note = self.activate(note)


    return note, tempo


class LyricsEncoder(nn.Module):
  @param(inp = ['LongTensor', (300, config.lyrics.L)])
  def __init__(self, vs):
    # vs: vocabulary size
    super().__init__()
    self.L, self.E, self.K, self.Co, self.M = config.lyrics.L, config.lyrics.E, config.lyrics.K, config.lyrics.Co, config.M
    self.dp = config.lyrics.dp
    assert self.L >= self.K, 'LyricsEncoder: Seq length should >= kernel size'

    self.embed = nn.Embedding(vs, self.E)
    self.dropout = nn.Dropout(self.dp, inplace=True)
    self.conv = nn.Conv2d(1, self.Co, (self.K, self.E))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(self.Co, self.M)
    self.activate = nn.Sigmoid()


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
    out = self.activate(out)

    return out

def translateWrap(translator):
  import lyrics2vec as l2v
  import dataset
  def translate(s):
    inp = l2v.convert(s)[:config.lyrics.L]
    inp = torch.Tensor(inp).unsqueeze(0)
    note, tempo = translator(inp)
    
    _, note = note.max(1)
    vec = note.data.squeeze().tolist(), tempo.data[0]
    out = dataset.vec2midi(vec)
    return out
  return translate

class Translator(nn.Module):
  @param(
    note = ['LongTensor', (300, config.music.L)], 
    tempo = ['Tensor', (300, )],
  )
  def __init__(self, init = None):
    # encoder: lyrics encoder / music encoder
    # decoder: music decoder
    super().__init__()
    self.L, self.E = config.music.L, config.music.E
    if init:
      encoder, decoder = init
      self.encoder = encoder
      self.decoder = decoder
    else:
      self.encoder = LyricsEncoder(vs)
      self.decoder = MusicDecoder()
      self.translate = translateWrap(self)
      self.train(False)

  def forward(self, inp):
    enc = self.encoder(inp)
    out = self.decoder(enc)
    return out 

  def wrapTar(self, tar):
    note, tempo = tar
    self.note.resize_(note.size()).copy_(note)
    self.tempo.resize_(tempo.size()).copy_(tempo)
    return Variable(self.note).view(-1), Variable(self.tempo)

if __name__ == '__main__':
  vsL, vsM = vs, config.note.size
  n = 3
  M = config.M
  lyr = torch.floor( torch.rand(n, config.lyrics.L) * vsL )
  note = torch.floor( torch.rand(n, config.music.L) * vsM )
  tempo = torch.floor( torch.rand(n) * vsM )
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
  assert outMd[0].size(0) == n * config.music.L
  assert outMd[0].size(1) == vsM
  assert outMd[1].size() == tempo.size()

  ae = Translator([me, md])
  print('Autoencoder: ', ae)
  outAe = ae(mus)
  assert outAe[0].size(0) == n * config.music.L
  assert outAe[0].size(1) == vsM
  assert outAe[1].size() == tempo.size()

  tr = Translator([le, md])
  print('Translator: ', tr)
  outTr = tr(lyr)
  assert outTr[0].size(0) == n * config.music.L
  assert outTr[0].size(1) == vsM
  assert outTr[1].size() == tempo.size()

