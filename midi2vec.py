import config

from collections import defaultdict
import csv

def diffTime(note):
  '''
  input: a TRACK of notes with E features, time is the absolute time
  output: as input but time is counted from the start of the last note
  '''
  time = config.music.feat2id['time']
  n = config.music.E
  diffs = [ f2[time]-f1[time] for f1,f2 in zip([[0]*n]+note[:-1], note) ]
  for i, diff in enumerate(diffs):
    note[i][time] = diff
  return note

def midiCsvEventReader(filename):
  with open(filename, 'r') as f:
    for record in csv.reader(f):
      record = [ s.strip() for s in record ]
      track, time, rtype = record[:3]
      time, track = int(time), int(track)
      value = [ int(s) for s in record[3:] if s.isdigit() ]
      yield track, rtype, time, value

def midiCsvReader(filename):
  '''
From midi-csv file to note and tempo.
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
  for track, rtype, time, value in midiCsvEventReader(filename):
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
  note = sorted(note, key=len)[-config.music.Ci:]
  return note, tempo

def chunk(note, tempo):
  note_snippets = defaultdict(list)
  for tid, track in enumerate(note):
    for time, track_snippet in chunkTrack(track):
      note_snippets[time].append(track_snippet)
  i = 0
  for time, note_snippet in note_snippets.items():
    if time >= tempo[i][0]:
      i += 1
    yield note_snippet, tempo[i][1]

def chunkTrack(note):
  '''
  input: a TACK of note
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

def midi2vec(midi):
  '''
  note, tempo = midiCsvReader(midi)
  for snippets in chunk(note, tempo):
  '''
