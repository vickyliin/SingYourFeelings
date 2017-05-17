from jseg.jieba import Jieba
jieba = Jieba()

def cut(s):
  return str(jieba.seg(s, pos=False))



def generateSegmentation(files="data/raw/*.txt", save_in="data/seg"):
  import glob, os

  if not os.path.exists(save_in):
    print("Make directory "+save_in)
    os.makedirs(save_in)
    
  files = list(glob.glob(files))
  print('Segmentalize {} files ...'.format(len(files)))
  for path in files:
    with open(path) as f:
      _s = f.read()
      seged = " ".join([cut(s) for s in _s.split()])
    save_path = os.path.join(save_in, os.path.basename(path))
    with open(save_path, "w") as f:
      f.write(seged) 
  print("Segmentalize finished")




def generateWordEmbedding(filenames):
  pass

