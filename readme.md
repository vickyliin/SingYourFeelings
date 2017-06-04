# 把你心情哼成歌

「[把你心情哼成歌](https://vickyliin.github.io/SingYourFeelings/gui/)」（Sing Your Feelings, SYF）是一個結合文字辨識和音樂生成的系統，把一段有意義的句子、詩詞或文章轉換成一段好聽的音樂，並能體現出這段文字的意涵。

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
- output/
    - {model name}.jsonl
    - {model name}-{id}.mid
 
- crawler/
- demo/
- gui/
    - index.html
    - ...
```

## Requirement

### Python3.5

Packages are listed in `requirement.txt`, so:
```bash
pip install -r requirement.txt
```

- PyTorch should be installed manually from the [website](http://pytorch.org/).

- [Jseg3](https://github.com/amigcamel/Jseg/tree/jseg3) should be installed by 

    ```bash
    pip install https://github.com/amigcamel/Jseg/archive/jseg3.zip
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

The dataset will be saved in `data/train.jsonl` and `data/valid.jsonl`.
    
### Train Model

```bash
python3 train.py
```

The trained parameters will be saved in `model/test.para`.

### Test Model

```bash
python3 test.py
```

The output midi files and input descriptions will be stored in the `output/` directory.

### Modification

The arguments are stored in `config.py`. One can find detailed descriptions about the arguments in it.

Other usage of the system can be found in the `demo/` directory.


