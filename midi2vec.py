from collections import namedtuple
import csv

from bisect import bisect_right as bs

import config

Tempo = namedtuple('Tempo', ['time', 'value'])
id = config.music.feat2id
min_pitch, max_pitch = config.note.pitch[0], config.note.pitch[-1]

def noteEmb(snippet):
  return list(filter(None, map(note2Id, snippet)))

def note2Id(note):
  '''
    input: ONE note with features
    output: note id
  '''
  if (0 < note[id['duration']] <= config.note.duration[-1]
      and min_pitch <= note[id['pitch']] <= max_pitch
      and not 0 < note[id['time']] < config.note.time[1]):
    note = tuple(d[bs(d, n)-1] for n, d in zip(note, config.note.divs))
    return config.note.note2id[note]
  else:
    return None

def chunk(midi2vec):
  def chunk(note, tempo):
    '''
    input: 
      note: note sequence in a midi file
      tempo: tempo sequence in a midi file
    output: (snippet, tempo)
      snippet: a snippet of notes whose feature time is diffTime.
      tempo: the tempo at which the snippet start
    '''
    T = config.music.T
    snippet, time, tid = [], 0, 0
    for feat in note:
      if feat[id['time']] > time:
        while time > tempo[tid].time and tid < len(tempo)-1:
          tid += 1
        if len(snippet) > T/2:
          yield diffTime(snippet), tempo[tid-1].value
        time += T
        snippet = []
      snippet.append(feat)
    '''
    if snippet:
      yield diffTime(snippet), tempo[tid-1].value
    '''

  def convert(filename):
    '''Convert a midi-csv file to music snippets.
    input: midi-csv file name
    output: snippet (note, tempo) list
      note: list of track
        track: list of note features
      tempo: int
    '''
    notes, tempos = midi2vec(filename)
    notes = sorted(notes, key=lambda feat: feat[id['time']])
    for chunk_notes, chunk_tempos in chunk(notes, tempos):
      snippet = {
        'note': noteEmb(chunk_notes),
        'tempo': chunk_tempos,
        'feature': dict(zip(config.music.id2feat, zip(*chunk_notes)))
      }
      yield snippet
      
  return convert

def diffTime(note):
  '''
  input: notes with absolute time
  output: as input but time is counted from the start of the last note
  '''
  if note:
    time = config.music.feat2id['time']
    n = config.music.E
    times = [feat[time] for feat in note]
    diffs = [0] + [t2-t1 for t1,t2 in zip(times[:-1], times[1:])]
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
  note, tempo, open_note = [], [], {}
  ppqn = 1 # pulses per beat
  for track, rtype, time, value in midiCsvReader(filename):
    time /= ppqn
    if rtype == 'Header':
      ppqn = value[-1]
    elif rtype == 'Tempo':
      tempo.append(Tempo(time, value[0]/config.tempo.default))
    elif rtype.startswith('Note'):
      channel, pitch, volume = value
      # close the note, add to a track in the note
      if (track, channel, pitch) in open_note:
        if volume == 0 or rtype == 'Note_off_c':
          feat = open_note.pop((track,channel,pitch))
          feat[id['duration']] = duration = time - feat[id['time']]
          note.append(feat)
      # open a new note
      elif rtype == 'Note_on_c':
        feat = [0]*len(config.music.id2feat)
        feat[id['pitch']] = pitch
        feat[id['volume']] = volume
        feat[id['time']] = time
        open_note[track,channel,pitch] = feat

  if len(tempo)==0:
    tempo.append(Tempo(0, 1))
  return note, tempo

