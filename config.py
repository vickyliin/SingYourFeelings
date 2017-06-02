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
    T: time period of a music sneppet
    L: max music length 
    Ci: music track number
    E: note feature number
    K: kernel size
    Co: output channel
    dp: dropout rate
  '''
 
  feat2id = {'pitch':0, 'time':1, 'duration':2, 'volume':3}
  id2feat = { v: k for k, v in feat2id.items() }
  L = 4 #300
  Ci = 2 #5
  E = len(feat2id)
  K = 2 #10
  Co = 200
  dp = 0.5
  T = 1000

class note(metaclass=Args):
  '''
  dim: embedding dimension
  size: number of embeddings
  '''
  from functools import reduce
  pitch = list(range(25,80))+[90]
  time = [0, 30, 48, 60, 96, 10000]
  duration = [0, 16, 30, 48, 60, 120, 3000]
  volumn = [0, 85, 150]
  divs = [pitch, time, duration, volumn]
  dim = 100
  size = reduce(lambda x,y: x*y, map(len, divs))

class tempo(metaclass=Args):
  scaler = 500000


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
M = 6 #100


