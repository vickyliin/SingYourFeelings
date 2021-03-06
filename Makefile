DATA_DIR = data
RAW_DIR = $(DATA_DIR)/raw
CSV_DIR = $(DATA_DIR)/csv
SEG_DIR = $(DATA_DIR)/seg
WORD2VEC = $(DATA_DIR)/word-vectors.txt
MIDICSV = midicsv
PYTHON = python
ICONV = iconv -f ISO-8859-1 -t utf-8 -o

MIDI_FILES = $(wildcard $(RAW_DIR)/*.mid)
TXT_FILES = $(wildcard $(RAW_DIR)/*.txt)
CSV_FILES = $(MIDI_FILES:$(RAW_DIR)/%.mid=$(CSV_DIR)/%.csv)

DATASET = $(DATA_DIR)/train.jsonl $(DATA_DIR)/valid.jsonl

all: dataset

crawl: | $(RAW_DIR)
	$(PYTHON) crawler/midi_download.py $(RAW_DIR)

midi2csv: $(CSV_FILES)

$(CSV_DIR) $(RAW_DIR) $(SEG_DIR):
	mkdir -p $@

TIMEOUT = timeout 9
$(CSV_FILES): $(CSV_DIR)/%.csv : $(RAW_DIR)/%.mid | $(CSV_DIR)
	-$(TIMEOUT) $(MIDICSV) "$<" | $(ICONV) "$@"

word2vec: $(WORD2VEC) 

$(WORD2VEC): $(TXT_FILES)
	$(PYTHON) lyrics2vec.py
	

dataset: $(DATASET)

$(DATASET): $(WORD2VEC)
	$(PYTHON) dataset.py

RM = rm -d
clean:
	$(RM) $(RAW_DIR) $(CSV_DIR) $(SEG_DIR)
