import config

from midiutil import MIDIFile
import pandas as pd
import numpy as np
import torch
import word2vec
import random

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
        self.n_batch = len(data['lyrics'])//batch_size+1
      if 'note' in data:
        getattr(self, name)['note'] = self.padNote(data)
        self.n_batch = len(data['note'])//batch_size+1

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
    begin = self.bs * i
    end = begin + self.bs
    if end >= len(self):
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
    data = list(self.data.values())
    data = list(zip(*data))
    random.shuffle(data)
    data = [ list(d) for d in zip(*data) ]
    for k, v in zip(self.data, data):
      self.data[k] = v

  def __len__(self):
    return self.n_batch
    

if __name__ == '__main__':
  from pprint import pprint
  from glob import glob
  from random import randrange
  L = config.lyrics.L
  n = 5
  lyrics = [ [randrange(vs) for _ in range(randrange(2*L)+1)] for _ in range(n) ]

  note = []
  L, E = config.music.L, config.music.E
  for _ in range(n):
    snippet = []
    for _ in range(config.music.Ci):
      track = [ [randrange(100) for _ in range(E)] for _ in range(randrange(2*L))]
      snippet.append(track)
    note.append(snippet)
  tempo = [ randrange(10000) for _ in range(n)]


  inp = {'lyrics': lyrics}
  tar = {'note': note, 'tempo': tempo}
  pprint(inp)
  pprint(tar)

  dataset = Dataset(inp, tar, 2)
