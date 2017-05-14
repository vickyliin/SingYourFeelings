class music:
  feat2id = {'pitch':0, 'time':1, 'duration':2, 'volume':3}
  id2feat = { v: k for k, v in feat2id.items() }
  # max music length
  L = 4 #300
  # music track number
  Ci = 2 #5
  # note feature number
  E = len(feat2id)
  # kernel size
  K = 2 #10
  # output channel
  Co = 200
  # dropout rate
  dp = .5

class lyrics:
  # max lyrics length
  L = 5 #500
  # word embedding size
  E = 200
  # kernel size (n-gram)
  K = 1
  # output channel (n-gram vector dim)
  Co = 250
  # dropout rate
  dp = .5
  # word2vec file
  lex = 'data/word-vectors.txt'

# encoded vector size
M = 6 #100



