import torch
from torch.autograd import Variable
import torch.nn as nn

use_cuda = torch.cuda.is_available()

def load(filename):
  if use_cuda:
    sd = torch.load(filename)
  else:
    sd = torch.load(filename, 
      map_location = lambda storage, loc: storage)

  return sd

class MusicEncoder(nn.Module):
  def _init__(self):
    supur().__init__()

    if use_cuda:
      self.cuda()

  def forward(self, inp):
    pass

class MusicDecoder(nn.Module):
  def __init__(self):
    super().__init__()

    if use_cuda:
      self.cuda()

  def forward(self, inp):
    pass


class Translator(nn.Module):
  def __init__(self):
    super().__init__()

    if use_cuda:
      self.cuda()

  def forward(self, inp):
    pass

  def translate(self, inp):
    pass
