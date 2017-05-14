from midiutil import MIDIFile
import pandas as pd
import torch
import config
import word2vec

lex = word2vec.load(config.lyrics.lex)

def lyr2vec(lyr):
  vec = torch.LongTensor(0)
  return vec

def midi2vec(midi):
  vec = torch.LongTensor(0)
  return vec

def vec2midi(vec):
  midi = MIDIFile(config.music.Ci)
  return midi

class Batch(tuple):
  def __len__(self):
    return self[0][0].size(0)

class Dataset:
  def __init__(self, filename):
    self.data = pd.read_json(DATA, lines = 1)

  def __getitem__(self, i):
    # get batch
    return Batch(inp, tar)

  def shuffle(self):
    pass

  def size(self):
    return len(self.data)

