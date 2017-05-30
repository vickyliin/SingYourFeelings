#!/usr/bin/env python
import model

import json
import pandas as pd
from os import makedirs


MODEL = 'test'
DATA = 'data/test.jsonl'
OUT_PATH = 'output/'
makedirs(OUT_PATH, exist_ok=True)
'''
{
  id: int,
  name: song name (str),
  lyrics: lyrics sequence vector (list int),
  note: note sequence matrix ( 3d list int),
  tempo: tempo (int),
}
'''
translator = model.Translator()

filename  = 'model/%s.para' % MODEL
sd = model.load(filename)
translator.load_state_dict(sd)

data = pd.read_json(DATA, lines=True)


fout = open('%s/%s.jsonl' % OUT_PATH, MODEL, 'w')
for id, name, inp in zip(data.id, data.name, data.lyrics):
  filename = '%s/%s-%s.midi' % (OUT_PATH, MODEL, id)
  print('Name  : '+name)
  print('Input : '+inp)
  print('Output: '+filename)

  out = translator.translate(inp)
  out.writeFile(filename)
  print('\n\n\n')

  print(json.dumps({
    'name': name,
    'input': inp,
    'output': filename,
  }, ensure_ascii=False), file=fout)



