import torch
from torch.autograd import Variable
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class param:
  def __init__(self, **args):
    self.args = args

  def __call__(init, fin):
    def __init__(self, *args, **kwargs):
      fin(self, *args, **kwargs)
      if use_cuda:
        self.cuda()
        for k, (t, v) in init.args.items():
          print(k, ':', t, v)
          setattr(self, k, getattr(torch,t)(*v).cuda())
      else:
        for k, v in init.args.items():
          print(k, ':', t, v)
          setattr(self, k, getattr(torch,t)(*v))

    return __init__

def load(filename):
  if use_cuda:
    sd = torch.load(filename)
  else:
    sd = torch.load(filename, 
      map_location = lambda storage, loc: storage)

  return sd

class MusicEncoder(nn.Module):
  # max music length
  L = 300
  # music channel number
  Ci = 10 
  # embedding size
  E = 50
  # CNN kernel size
  K = 10
  # CNN output channel
  Co = 200

  # encoded vector size
  M = 100

  # Dropout rate
  dp = .5

  @param(inp = ['LongTensor', (300, Ci, L)])
  def __init__(self, vs):
    # vs: vocabulary size
    super().__init__()
    self.embed = nn.Embedding(vs, self.E)
    self.dropout = nn.Dropout(self.dp)
    self.conv = nn.Conv2d(self.Ci, self.Co, (self.K, self.E))
    self.pool = nn.MaxPool1d(self.L, ceil_mode=True)
    self.linear = nn.Linear(self.Co, self.M)


  def forward(self, inp):
    # inp: datatype torch tensor, N x Ci x L
    assert type(inp).__name__.endswith('Tensor')
    inp = self.inp.resize_(inp.size()).copy_(inp)
    inp = Variable(inp)
    emb = self.embed(inp.view(-1, self.L))        # N*Ci x L x E
    emb = emb.view(-1, self.Ci, self.L, self.E)   # N x Ci x L x E
    #emb = self.dropout(emb)
    hid = self.conv(emb).squeeze(-1)              # N x Co x L-
    hid = self.pool(hid).squeeze(-1)              # N x Co

    out = self.linear(hid)                        # N x M

    return out


class MusicDecoder(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, inp):
    pass


class Translator(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, inp):
    pass

  def translate(self, inp):
    pass
