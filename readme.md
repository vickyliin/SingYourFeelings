# 把你心情哼成歌

「把你心情哼成歌」（Sing Your Feelings, SYF）是一個結合文字辨識和音樂生成的系統，把一段有意義的句子、詩詞或文章轉換成一段好聽的音樂，並能體現出這段文字的意涵。

## Files

- crawler
- data
    - raw
    - train.jsonl
    - test.jsonl
- model
    - {model name}.para
- output
    - {model name}.jsonl
    - {model name}-{id}.midi
- platform

- dataset.py
- model.py
- test.py
- train.py


## Requirement

- python3.5
- numpy
- MIDIUtil
- torch
- pandas

### MIDIUtil

```python=
from midiutil import MIDIFile

degrees  = [60, 62, 64, 65, 67, 69, 71, 72]  # MIDI note number
track    = 0
channel  = 0
time     = 0    # In beats
duration = 1    # In beats
tempo    = 60   # In BPM
volume   = 100  # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
                      # automatically)
MyMIDI.addTempo(track, time, tempo)

for i, pitch in enumerate(degrees):
    MyMIDI.addNote(track, channel, pitch, time + i, duration, volume)

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)
```
