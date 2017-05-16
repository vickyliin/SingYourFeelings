# 把你心情哼成歌

「把你心情哼成歌」（Sing Your Feelings, SYF）是一個結合文字辨識和音樂生成的系統，把一段有意義的句子、詩詞或文章轉換成一段好聽的音樂，並能體現出這段文字的意涵。

## Files
```
- crawler/
    - midi.py
    - midi_downloader.py
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
- platform/

- dataset.py
- model.py
- test.py
- train.py
- Makefile
- requirement.txt
```

## Requirement

- python3.5: packages listed in requirement.txt
    ```bash
    pip install -r requirement.txt
    ```
    > pytorch shoould be installed manually from the [website](http://pytorch.org/)

- [midicsv](http://www.fourmilab.ch/webtools/midicsv/)
    ```bash
    wget http://www.fourmilab.ch/webtools/midicsv/midicsv-1.1.tar.gz
    tar zxcf midicsv-1.1.tar.gz
    cd midicsv-1.1
    make
    make install INSTALL_DEST=path_to_install
    ```
