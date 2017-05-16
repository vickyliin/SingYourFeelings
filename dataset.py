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
  def __init__(self, data, batch_size, pad_value):
    '''
    data: dict of (lyrics, note, tempo)
      lyrics: list, lyrics vector
      note: list, note matrix of the music snippet
      tempo: int, tempo of the music snippet
    '''
    self.bs = batch_size
    self.data = data

    if 'lyrics' in self.data:
      self.padLyrics(pad_value)
    if 'note' in self.data:
      self.padNote()

  def padNote(self):
    def padNote(note):
      E, L = config.music.E, config.music.L
      for track in note:
        if len(track) < L:
          yield track + [ [0]*E for _ in range(L-len(track)) ]
        else:
          yield track[:L]

    self.data['note'] = [ list(padNote(note)) for note in self.data['note'] ]

  def padLyrics(self, pv):
    def padLyrics(lyr):
      L = config.music.L
      if len(lyr) < L:
        return lyr + [pv]*( L-len(lyr) )
      else:
        return lyr[:L]

    self.data['lyrics'] = [ padLyrics(lyrics) for lyrics in self.data['lyrics'] ]

  def __getitem__(self, i):
    begin = self.bs * i
    end = begin + self.bs
    if end >= len(self.data):
      end = None

    print(begin, end)
    lyrics, note, tempo = None, None, None
    if 'lyrics' in self.data:
      lyrics = torch.Tensor(self.data['lyrics'][begin:end])
    if 'note' in self.data:
      note = torch.Tensor(self.data['note'][begin:end])
      tempo = torch.Tensor(self.data['tempo'][begin:end])

    return lyrics, note, tempo

  def shuffle(self):
    data = list(self.data.values())
    data = list(zip(*data))
    random.shuffle(data)
    data = [ list(d) for d in zip(*data) ]
    for k, v in zip(self.data, data):
      self.data[k] = v

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


  data = {'lyrics': lyrics, 'note': note, 'tempo': tempo}
  pprint(data)


  dataset = Dataset(data, 2, 0)

