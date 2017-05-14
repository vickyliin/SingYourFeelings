import model

import json
import pandas as pd


MODEL = 'test'
DATA = 'data/test.jsonl'
OUT_PATH = 'output/'
'''
{
  id: int,
  raw: [lyrics (str), music path (str)],
  lyrics: lyrics sequence vector (list float),
  music: music sequence matrix (list list float)
}
'''
translator = model.Translator()

filename  = 'model/%s.para' % MODEL
sd = model.load(filename)
translator.load_state_dict(sd)

data = pd.read_json(DATA, lines = 1)


fout = open('%s/%s.jsonl' % OUT_PATH, MODEL, 'w')
for id, (inp, tar) in zip(data.id, data.raw):
  filename = '%s/%s-%s.midi' % (OUT_PATH, MODEL, id)
  print('Output: '+filename)
  print('Target: '+tar)
  print('Input : \n'+inp)

  out = translator.translate(inp)
  out.writeFile(filename)
  print('\n\n\n')

  print(json.dumps({
    'output': filename,
    'target': tar,
    'input': inp
  }, ensure_ascii=False), file=fout)



