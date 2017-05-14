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
