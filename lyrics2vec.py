#!/usr/bin/env python3
from jseg.jieba import Jieba
import glob, os

def cut(s):
  return str(jieba.seg(s, pos=False))



def generateSegmentation(files="data/raw/*.txt", save_in="data/seg"):
  global jieba
  jieba = Jieba()

  if not os.path.exists(save_in):
    print("Make directory "+save_in)
    os.makedirs(save_in)
    
  files = list(glob.glob(files))
  print('Segmentalizing {} files ...'.format(len(files)))
  for path in files:
    with open(path) as f:
      _s = f.read()
      seged = " ".join([cut(s) for s in _s.split()])
    save_path = os.path.join(save_in, os.path.basename(path))
    with open(save_path, "w") as f:
      f.write(seged) 
  print("Segmentalize finished")

def generateWordEmbedding(files="data/seg/*.txt", save_as="data/word-vector.bin"):
  import word2vec
  corpus_path = "word2vec-corpus.tmp"
  with open(corpus_path, "w") as corpus:
    for path in glob.glob(files):
      with open(path) as f:
        corpus.write(f.read()+"\n")
  word2vec.word2vec(corpus_path, save_as, size=100, verbose=True)
  os.remove(corpus_path)
  
if __name__=="__main__":
  print("Will segmentalize/make word embedding for files in data/raw/*.txt")
  generateSegmentation()
  generateWordEmbedding()
  print("Done")

