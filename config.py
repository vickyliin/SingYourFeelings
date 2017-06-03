import yaml

class Args(type):
  def __repr__(self):
    args = { k: v for k, v in vars(self).items() if not k.startswith('_') }
    if self.__doc__:
      args = {'Description': yaml.load(self.__doc__), 'Data':args}
    s = yaml.dump({self.__name__: args}, default_flow_style=False)
    return s

class autoencoder(metaclass=Args):
  optim = 'Adam'
  optim_args = dict(
    lr = 3e-3
  )
  batch_size = 30
  max_epoch = 10000
  endure = 20
  loss_cate = 'CrossEntropyLoss'
  loss_val = 'MSELoss'

class translator(metaclass=Args):
  name = 'test'
  optim = 'RMSprop'
  optim_args = dict(
    lr = 3e-3
  )
  batch_size = 30
  max_epoch = 10000
  endure = 30
  loss_cate = 'CrossEntropyLoss'
  loss_val = 'MSELoss'

class music(metaclass=Args):
  '''
    T: time period of a music sneppet (in beat)
    L: max music length 
    Ci: music track number
    E: note feature number
    K: kernel size
    Co: output channel
    dp: dropout rate
  '''
 
  feat2id = {'pitch':0, 'time':1, 'duration':2, 'volume':3}
  id2feat = { v: k for k, v in feat2id.items() }
  L = 10 #300
  Ci = 100 #5
  E = len(feat2id)
  K = 3 #10
  Co = 1000
  dp = 0.25
  T = 30

class note(metaclass=Args):
  '''
  dim: embedding dimension
  size: number of embeddings
  '''
  from itertools import product
  pitch = list(range(25,100))
  duration = [1/4, 1/3, 1/2, 1, 1.5, 2, 3, 4]
  time = [0] + duration + [6, 8, 10, 12, 16]
  volume = [85, 150]

  divs = [pitch, time, duration, volume]
  note2id = {k: i for i, k in enumerate(product(*divs), 1)}
  note2id[0,0,0,0] = 0
  id2note = {v: k for k, v in note2id.items()}
  size = len(note2id)

  dim = 30

class tempo(metaclass=Args):
  default = 500000 # mus/beat


class lyrics(metaclass=Args):
  '''
    L: max lyrics length
    E: word embedding size
    K: kernel size (n-gram)
    Co: output channel (n-gram vector dim)
    dp: dropout rate
    lex: word2vec file
  '''
  L = 5 #500
  E = 200
  K = 1
  Co = 250
  dp = 0.8
  lex = 'data/word-vectors.txt'

# encoded vector size
M = 200 #100


