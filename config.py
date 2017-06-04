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
  endure = 1000
  loss_cate = 'CrossEntropyLoss'
  loss_val = 'MSELoss'

class music(metaclass=Args):
  '''
    T: time period of a music sneppet (in beat)
    L: max music length 
    E: note feature number
    K: kernel size
    dp: dropout rate
  '''
 
  id2feat = 'pitch time duration volume'.split()
  feat2id = {feat: id for id, feat in enumerate(id2feat)}
  L = 10
  E = len(feat2id)
  K = 3
  Co = 500
  dp = 0.5
  T = 32

class note(metaclass=Args):
  '''
  dim: embedding dimension
  size: number of embeddings
  '''
  from itertools import product
  pitch = list(range(36,90))
  duration = [x/2 for x in range(1, 4)]
  time = [x/16 for x in range(8)]
  time += [x/8 for x in range(4, 8)]
  time += [x/2 for x in range(2, 4)]
  volume = [80, 100]

  divs = [pitch, time, duration, volume]
  note2id = {k: i for i, k in enumerate(product(*divs), 1)}
  note2id[0,0,0,0] = 0
  id2note = {v: k for k, v in note2id.items()}
  size = len(note2id)
  print(size)

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
  L = 10 #500
  E = 200
  K = 1
  Co = 200
  dp = 0.05
  lex = 'data/word-vectors.txt'

# encoded vector size
M = 200 #100


