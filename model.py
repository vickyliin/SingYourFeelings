import logging
logger = logging.getLogger('model')

import torch
from torch.autograd import Variable
import torch.nn as nn

'''
from vectorize import vectorize
from featurize import featurize
'''

use_cuda = torch.cuda.is_available()

class Decoder(nn.Module):
  def __init__(self, emb_size, hidden_size, **kwargs):
    super().__init__()
    es = emb_size
    hs = hidden_size

    # self.linear = nn.Linear(2*hs, hs)
    # self.activate = nn.Tanh()

    rnn = getattr(nn, kwargs['model'])
    self.rnn = rnn(es, hs, dropout=kwargs['dropout'])

    if use_cuda:
      self.cuda()

  def forward(self, inputs, h0):
    # b: Li x H (in_hidden)
    # out_embbed: Lo x D
    b, out_emb = inputs

    # Wb = self.linear(b)              # Li x H
    # s = self.activate(s)

    out_emb = out_emb.unsqueeze(1)   # Lo x 1 x D
    s, h1 = self.rnn(out_emb, h0)    # Lo x 1 x H
    s = s.squeeze(1)                 # Lo x H

    # Attention
    alpha = s @ b.t()
    alpha = nn.Softmax()(alpha)      # Lo x Li
    c = alpha @ b                    # Lo x H

    # oup = torch.cat([s, c], 1)       # Lo x 2H
    oup = s+c

    return oup, h1



class MusicLM(nn.Module):
  def __init__(self, **kwargs):
    # hidden_size: word embedding & RNN size
    # w: word, n: note
    super().__init__()
    nvs, wvs = kwargs['note_vocab_size'], kwargs['word_vocab_size']
    nes = kwargs.setdefault('note_emb_size', 200)
    wes = kwargs.setdefault('word_emb_size', 200)
    hs = kwargs.setdefault('hidden_size', 200)

    dp = kwargs.setdefault('dropout', .5)
    model = kwargs.setdefault('model', 'GRU')
    self.create_args = kwargs

    self.wemb = nn.Embedding(wvs, wes)
    self.nemb = nn.Embedding(nvs, nes)

    rnn = getattr(nn, model)
    self.encoder = rnn(wes, hs, dropout=dp, bidirectional=True)
    self.decoder = Decoder(nes, 2*hs, dropout=dp, model=model)

    self.pointer = nn.Linear(2*hs, nvs)
    self.lex = None

    self.initParams()

    if use_cuda:
      self.cuda()

  def initParams(self):
    hs = self.create_args['hidden_size']
    self.register_buffer('h0', torch.zeros(2, 1, hs))
    if self.create_args['model'] == 'LSTM':
      self.register_buffer('c0', torch.zeros(2, 1, hs))

    self.inp = torch.LongTensor(100)
    self.tar = torch.LongTensor(100)
    if use_cuda:
      self.inp = self.inp.cuda()
      self.tar = self.tar.cuda()


  def initForward(self, inp, tar=None):
    h0 = Variable(self.h0, requires_grad=False)
    if self.create_args['model'] == 'LSTM':
      c0 = Variable(self.c0, requires_grad=False)
      h0 = (h0, c0)

    inp = Variable(self.inp.resize_(inp.size()).copy_(inp))
    if tar is not None:
      tar = Variable(self.tar.resize_(tar.size()).copy_(tar))
      inp = (inp, tar)

    return h0, inp

  def forward(self, inputs):
    # inputs type: torch.LongTensor 

    in_sent, out_sent = inputs

    h0, (inp, tar) = self.initForward(in_sent, out_sent)

    inp = self.wemb(inp).unsqueeze(1) # Li x 1 x Di
    h, h1 = self.encoder(inp, h0)     # Li x 1 x 2H, tuple

    if type(h1) == tuple:
      h1 = tuple( h.view(1,1,-1) for h in h1 )
    else:
      h1 = h1.view(1,1,-1)
    h = h.squeeze(1)

    tar = self.nemb(tar)              # Lo x D
    oup, _ = self.decoder((h, tar), h1)

    oup = oup.squeeze(1)
    oup = self.pointer(oup)

    return oup

  def generate_(self, inp, hint=None, beam = 1):
    if beam != 1:
      logger.warning('Beam search havn\'t been implmented.')


    if hint:
        bos = torch.LongTensor([self.lex.BOS]+hint)
    else:
        bos = torch.LongTensor([self.lex.BOS])
    h0, (inp, bos) = self.initForward(inp, bos)

    inp = self.wemb(inp).unsqueeze(1)
    h, h1 = self.encoder(inp, h0)
    if type(h1) == tuple:
      h1 = tuple(h.view(1,1,-1) for h in h1)
    else:
      h1 = h1.view(1,1,-1)
    h = h.squeeze(1)
    #print(h1[1])

    seq = [bos]
    for pos in range(30):
      tar = self.nemb(seq[-1])
      oup, h1 = self.decoder((h,tar), h1)

      oup = oup.squeeze(1)
      oup = self.pointer(oup)

      maxvalue, idx = oup.squeeze(0).max(0)

      seq.append(idx)

      if idx.data[0] == self.lex.EOS:
        break

    return torch.cat(seq)

'''
  def respond(self, seq):
    assert self.lex, \
      'Lexicon needed. Assign the lexicon to attribute "lex".'
    seq = featurize([seq])
    inp = vectorize(seq['word'][0], self.lex)
    #print("inp:", inp)

    inp = torch.LongTensor(inp[1:-1])

    oup = self.generate_(inp)
    #print("out:", list(oup.data))
    words = [ self.lex.vocab[x] for x in oup.data[1:-1] ]

    return ' '.join(words)
'''


