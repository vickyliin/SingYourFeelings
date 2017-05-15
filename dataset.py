import config

from midiutil import MIDIFile
import pandas as pd
import torch
import word2vec
import csv
from collections import defaultdict

lex = word2vec.load(config.lyrics.lex)

def lyr2vec(lyr):
  vec = torch.LongTensor(0)
  return vec

def padNote(note):
  m = max(len(track) for track in note.values())
  n = config.music.E
  note = [ track + [ [0]*n for _ in range( m-len(track) ) ]\
    for track in note.values() if track ]
  return note

def midiCsvReader(filename):
  with open(filename, 'r') as f:
    for record in csv.reader(f):
      record = [ s.strip() for s in record ]
      track, time, rtype = record[:3]
      time, track = int(time), int(track)
      value = [ int(s) for s in record[3:] if s.isdigit() ]
      yield track, rtype, time, value

def midi2vec(midi):
  '''
From midi-csv file to torch tensor.
input:
  midi: midi-csv filename
output:
  note: 3d torch float tensor, Ci x L x E
  tempo: 2d torch float tensor, Lt x [time, value]
  '''
  id = config.music.feat2id
  note, tempo, open_note = defaultdict(list), [], {}
  for track, rtype, time, value in midiCsvReader(midi):
    if rtype == 'Tempo':
      tempo.append([time, value[0]])
    elif rtype.startswith('Note'):
      channel, pitch, volume = value
      # close the note, add to a track in the note
      if (track, channel, pitch) in open_note:
        if volume == 0 or rtype == 'Note_off_c':
          feat = open_note.pop((track,channel,pitch))
          feat[id['duration']] = time - feat[id['time']]
          note[track, channel].append(feat)
      # open a new note
      elif rtype == 'Note_on_c':
        feat = [0]*len(id)
        feat[id['pitch']] = pitch
        feat[id['volume']] = volume
        feat[id['time']] = time
        open_note[track,channel,pitch] = feat

  note = padNote(note)
  return torch.Tensor(note), torch.Tensor(tempo)

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

