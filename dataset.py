#!/usr/bin/env python
import config

from midiutil import MIDIFile
import pandas as pd
import torch
import word2vec
import random, yaml
import lyrics2vec as l2v
import midi2vec as m2v

lex = word2vec.load(config.lyrics.lex)
vs = len(lex.vocab)

def lyr2vec(lyr):
  vec = torch.LongTensor(0)
  return vec

def vec2midi(vec):
  midi = MIDIFile(config.music.Ci)
  return midi

class Dataset:
  def __init__(self, inp, tar, batch_size, pad_value=0):
    '''
    inp, tar: dict of (lyrics, note, tempo)
      lyrics: list, lyrics vector
      note: list, note matrix of the music snippet
      tempo: int, tempo of the music snippet
    '''
    self.bs = batch_size
    self.inp = inp
    self.tar = tar

    for name, data in [('inp',inp), ('tar',tar)]:
      if 'lyrics' in data:
        getattr(self, name)['lyrics'] = self.padLyrics(data, pad_value)
        self.size = len(data['lyrics'])
      if 'note' in data:
        getattr(self, name)['note'] = self.padNote(data)
        self.size = len(data['note'])

  def padNote(self, data):
    def padNote(note):
      E, L = config.music.E, config.music.L
      for track in note:
        if len(track) < L:
          yield track + [ [0]*E for _ in range(L-len(track)) ]
        else:
          yield track[:L]

    return [ list(padNote(note)) for note in data['note'] ]

  def padLyrics(self, data, pv):
    def padLyrics(lyr):
      L = config.music.L
      if len(lyr) < L:
        return lyr + [pv]*( L-len(lyr) )
      else:
        return lyr[:L]

    return [ padLyrics(lyrics) for lyrics in data['lyrics'] ]

  def __getitem__(self, i):
    if i >= len(self):
      raise IndexError('Index %d out of range (%d)' % (i, len(self)-1))
    begin = self.bs * i
    end = begin + self.bs
    if end >= self.size:
      end = None

    lyrics, note, tempo = None, None, None
    pair = [(),()]
    for ii, data in [(0,self.inp), (1,self.tar)]:
      if 'lyrics' in data:
        pair[ii] = torch.Tensor(data['lyrics'][begin:end])
      if 'note' in data:
        pair[ii] += (torch.Tensor(data['note'][begin:end]),)
        pair[ii] += (torch.Tensor(data['tempo'][begin:end]),)

    return tuple(pair)

  def shuffle(self):
    data = list(self.inp.values())+list(self.tar.values())
    data = list(zip(*data))
    random.shuffle(data)
    data = [ list(d) for d in zip(*data) ]
    for k, v in zip(self.inp, data[:len(self.inp)]):
      self.inp[k] = v
    for k, v in zip(self.tar, data[len(self.inp):]):
      self.tar[k] = v

  def __len__(self):
    return self.size//self.bs + 1

  def __repr__(self):
    return yaml.dump({'Dataset': dict(
      data = dict(
        inp = ', '.join(list(self.inp)),
        tar = ', '.join(list(self.tar)),
      ),
      size = self.size,
      batch_size = self.bs,
    )}, default_flow_style=False)

  def split(self, ratio=0.2):
    n = int(self.size*(1-ratio))
    def div(d, is_f):
      r = {}
      for k, v in d.items():
        r[k] = v[:n] if is_f==0 else v[n:]
      return r
    d1, d2 = [Dataset(div(self.inp,i), div(self.tar,i), self.bs)
        for i in [0, 1]]
    return d1, d2
    
def loadData(n=0, lyr_path="data/seg/*.txt", midi_path="data/csv/*.csv"):
  import glob, os
  name = lambda x: os.path.splitext(os.path.basename(x))[0]
  lyr_path, midi_path = list(glob.glob(lyr_path)), list(glob.glob(midi_path))
  both = set(map(name, lyr_path)).intersection(set(map(name, midi_path)))
  lyr_path  = list(filter(lambda n:name(n) in both, lyr_path))
  midi_path = list(filter(lambda n:name(n) in both, midi_path))
  lyr_path.sort( key=lambda f: os.path.basename(f))
  midi_path.sort(key=lambda f: os.path.basename(f))
  if n==0: n = len(lyr_path)
  for lyr, midi in list(zip(lyr_path, midi_path))[:n]:
    yield lyr, midi

def loadDataset(n=0):
  from random import randrange
  from collections import defaultdict
  L = config.lyrics.L
  n = 50

  lyrics, note, tempo = [], [], []

  print("Loading training data ...")
  for lyrics_path, midi_path in loadData(n):
    lyr = l2v.convert(lyrics_path)
    for n, t in m2v.convert(midi_path):
      lyrics.append(lyr)
      note.append(n)
      tempo.append(t)
  print("Done")

  inp = {'lyrics': lyrics}
  tar = {'note': note, 'tempo': tempo}

  class Note:
    def __init__(self, data):
      self.data = torch.Tensor(data)
    def __eq__(self, other):
      return torch.equal(self.data, other.data)

  dataset_tr = Dataset(inp, tar, 2)
  lyr2note = defaultdict(list)
  for k, v in zip(dataset_tr.inp['lyrics'], dataset_tr.tar['note']):
    lyr2note[tuple(k)].append(Note(v))

  dataset_tr.shuffle()
  for i in range(dataset_tr.size):
    k = tuple(dataset_tr.inp['lyrics'][i])
    v = Note(dataset_tr.tar['note'][i])
    assert v in lyr2note[k], '\nout: {}\ntar: {}'.format(v, lyr2note[k])

  dataset_ae = Dataset(tar, tar, 2)

  return dataset_ae, dataset_tr

