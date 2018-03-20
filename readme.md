# 把你心情哼成歌

「把你心情哼成歌」（Sing Your Feelings, SYF）是一個結合文字辨識和音樂生成的系統，把一段有意義的句子、詩詞或文章轉換成一段好聽的音樂，並能體現出這段文字的意涵。

## Files
```
- data/
    - raw/
        - *.mid
        - *.txt
    - csv/
        - *.csv
    - train.jsonl
    - test.jsonl
    - word-vectors.txt
- model/
    - {model name}.para
 
- crawler/
- gui/
    - index.html
    - ...
```

## Requirement

### Python3.5

- PyTorch should be installed manually from the [website](http://pytorch.org/).

- [Jseg3](https://github.com/amigcamel/Jseg/tree/jseg3) should be installed by 

    ```bash
    pip install https://github.com/amigcamel/Jseg/archive/jseg3.zip
    ```

Other packages are listed in `requirement.txt`, so:

```bash
pip install -r requirement.txt
```

### [midicsv](http://www.fourmilab.ch/webtools/midicsv/)

```bash
wget http://www.fourmilab.ch/webtools/midicsv/midicsv-1.1.tar.gz
tar zxvf midicsv-1.1.tar.gz
cd midicsv-1.1
make
make install INSTALL_DEST=path_to_install
```

## Usage

### Prepare Data

```bash
make
```

This will generate in the `data/` folder: 

- the word vectors `word-vectors.txt`

- the dataset `train.jsonl` and `valid.jsonl`

### Train Model

```bash
python3 train.py
```

The trained parameters will be saved in `model/test.para`.

### Server

```
python3 server.py
```

## Modification

### Train

The arguments are stored in `config.py`. One can find detailed descriptions about the arguments in it.

### Server

- You can choose your own model, port and cuda usage in `server.py`.

    ```python
    para = 'model/test.para' # path to your trained model
    model.use_cuda = False
    port = 80
    ```
