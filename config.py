class music:
  # max music length
  L = 4 #300
  # music channel number
  Ci = 2 #5
  # note embedding size
  E = 50
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

# encoded vector size
M = 6 #100



