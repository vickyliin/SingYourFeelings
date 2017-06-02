import config

from collections import defaultdict
import csv

def noteEmb(snippet):
  return [note2Id(note) for note in snippet if note2Id(note) is not None]
def note2Id(note):
  from bisect import bisect_left as bs
  if note[2]==0: # duration=0
    return None
  divs = config.note.divs
  idxs = [ min(bs(d, n), len(d)-1) for n, d in zip(note, divs) ]
  idx, base = 0, 1
  for i, d in zip(idxs, divs):
    idx += i*base
    base *= len(d)
  return idx
def id2Note(idx):
  divs = config.note.divs
  note = []
  for d in divs:
    note.append(d[idx%len(d)])
    idx //= len(d)
  return note

def chunk(fin):
  def chunk(note, tempo):
    snippets = defaultdict(list)
    for track in note:
      for time, track_snippet in chunkTrack(track):
        snippets[time].append(track_snippet)
    i = 0
    for time, snippet in snippets.items():
      while time > tempo[i][0] and i < len(tempo)-1:
        i += 1
      snippet = list(map(noteEmb, snippet))
      yield snippet, tempo[i-1][1]/config.tempo.scaler

  def convert(filename):
    '''Convert a midi-csv file to music snippets.
    input: midi-csv file name
    output: snippet (note, tempo) list
      note: list of track
        track: list of note features
      tempo: int
    '''
    id = config.music.feat2id
    note, tempo = fin(filename)
    note = sorted(note, key=len)[-config.music.Ci:]
    note = [ sorted(track, 
      key = lambda feat: feat[id['time']]) \
      for track in note]
    return list(chunk(note, tempo))

  return convert

def diffTime(note):
  '''
  input: a TRACK of notes with E features, time is the absolute time
  output: as input but time is counted from the start of the last note
  '''
  time = config.music.feat2id['time']
  n = config.music.E
  times = [ feat[time] for feat in note ]
  diffs = [ t2-t1 for t1,t2 in zip([0]+times[:-1], times) ]
  for i, diff in enumerate(diffs):
    note[i][time] = diff
  return note

def midiCsvReader(filename):
  with open(filename, 'r') as f:
    for record in csv.reader(f):
      record = [ s.strip() for s in record ]
      track, time, rtype = record[:3]
      time, track = int(time), int(track)
      value = [ int(s) for s in record[3:] if s.isdigit() ]
      yield track, rtype, time, value

@chunk
def convert(filename):
  '''
Convert midi-csv file to note and tempo.
input:
  midi: midi-csv filename
output:
  note: list of Ci tracks with maximum notes
    track: list of notes with E features
  tempo: list of [time, value]
    time: int, the time at which the tempo changes
    value: int, the value of tempo
  '''
  id = config.music.feat2id
  note, tempo, open_note = defaultdict(list), [], {}
  for track, rtype, time, value in midiCsvReader(filename):
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

  note = [ track for track in note.values() ]
  if len(tempo)==0:
    tempo.append((0, 500000))
  return note, tempo

def chunkTrack(note):
  '''
  input: a TRACK of note
  output: (time, snippet)
    time: the time at which the snippet start
    snippet: a snippet of notes whose feature time is diffTime.
  '''
  T = config.music.T
  time = config.music.feat2id['time']
  snippet, end = [], T
  for feat in note:
    if feat[time] > end:
      yield end-T, diffTime(snippet)
      end += T
      snippet = []
    snippet.append(feat)

