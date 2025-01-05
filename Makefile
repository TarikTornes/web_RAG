# Define the scripts directory
SCRIPT_DIR = scripts

# Define a list of specific files (without the directory part)
FILES = extract_webtext.sh run_all.sh setup.sh webpage_retrieval.sh

# Default target that applies chmod +x to each specified file in the scripts directory
all:
	@for file in $(FILES); do \
		chmod +x $(SCRIPT_DIR)/$$file; \
	done

