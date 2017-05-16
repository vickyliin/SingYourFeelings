DATA_DIR = data
RAW_DIR = $(DATA_DIR)/raw
CSV_DIR = $(DATA_DIR)/csv
MIDICSV = midicsv
PYTHON = python

MIDI_FILES = $(wildcard $(RAW_DIR)/*.mid)
CSV_FILES = $(MIDI_FILES:$(RAW_DIR)/%.mid=$(CSV_DIR)/%.csv)


crawl: | $(RAW_DIR)
	$(PYTHON) crawler/midi_download.py $(RAW_DIR)

midi2csv: $(CSV_FILES)

$(CSV_DIR) $(RAW_DIR):
	mkdir -p $@

$(CSV_FILES): $(CSV_DIR)/%.csv : $(RAW_DIR)/%.mid | $(CSV_DIR)
	-$(MIDICSV) "$<" "$@"


RM = rm -d
clean:
	$(RM) $(RAW_DIR) $(CSV_DIR)
