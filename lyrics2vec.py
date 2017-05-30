#!/usr/bin/env python3
from jseg.jieba import Jieba
import glob, os
import word2vec
import numpy as np

jieba = None
UNK = '<UNK>'
def cut(s):
  global jieba
  if jieba is None:
    jieba = Jieba()
  return sum([list(jieba.seg(_s, pos=False).raw) for _s in s.split()], [])

def genSegment(files='data/raw/*.txt', save_in='data/seg'):

  if not os.path.exists(save_in):
    print('Make directory '+save_in)
    os.makedirs(save_in)
    
  files = list(glob.glob(files))
  print('Segmentalizing {} files ...'.format(len(files)))
  for path in files:
    with open(path) as f:
      seged = ' '.join(cut(f.read()))
    save_path = os.path.join(save_in, os.path.basename(path))
    with open(save_path, 'w') as f:
      f.write(seged) 
  print('Segmentalize finished')

def genWordEmbedding(files='data/seg/*.txt', save_as='data/word-vectors.txt'):
  corpus_path = 'word2vec-corpus.tmp'
  with open(corpus_path, 'w') as corpus:
    for path in glob.glob(files):
      with open(path) as f:
        corpus.write(f.read()+'\n')
    corpus.write((" "+UNK)*10)
  word2vec.word2vec(corpus_path, save_as, size=100, verbose=True, binary=0)
  w2v = word2vec.load(save_as)
  # set UNK's vector to mean of all vectors
  w2v.vectors[w2v.vocab_hash[UNK]] = np.mean(w2v.vectors, axis=0)
  with open(save_as, "w") as corpus: # w2v.save
    corpus.write("%d %d\n" % w2v.vectors.shape)
    for i in range(w2v.vectors.shape[0]):
      print(w2v.vocab[i], *(str(v)[:9] for v in w2v.vectors[i]), file=corpus)
  os.remove(corpus_path)
  

w2v = None
def convert(lyrics, usingW2V="data/word-vectors.txt", is_file=False):
  """Convert lyrics to word id list.
  input:
    lyrics: input str or file name
  """
  global w2v
  if is_file: # file
    with open(lyrics) as f:
      lyrics = f.read()
    lyrics = lyrics.split()
  else:
    terms = cut(lyrics)
  if w2v is None:
    w2v = word2vec.load(usingW2V)

  unk_id = w2v.vocab_hash[UNK]
  return [w2v.vocab_hash.get(t, unk_id) for t in lyrics]

if __name__=='__main__':
  print('Will segmentalize/make word embedding for files in data/raw/*.txt')
  genSegment()
  genWordEmbedding()
  print('Done')

