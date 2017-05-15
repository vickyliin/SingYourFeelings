DATA_DIR = data
RAW_DIR = $(DATA_DIR)/raw
CSV_DIR = $(DATA_DIR)/csv
MIDICSV = midicsv

MIDI_FILES = $(wildcard $(RAW_DIR)/*.mid)
LYR_FILES = $(MIDI_FILES:%.mid=%.txt)
CSV_FILES = $(MIDI_FILES:$(RAW_DIR)/%.mid=$(CSV_DIR)/%.csv)


midi2csv: $(CSV_FILES)

$(CSV_DIR):
	mkdir -p $@

$(CSV_FILES): $(CSV_DIR)/%.csv : $(RAW_DIR)/%.mid $(RAW_DIR)/%.txt | $(CSV_DIR)
	-$(MIDICSV) $< $@
