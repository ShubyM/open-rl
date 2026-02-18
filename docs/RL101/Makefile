# Generate SVG and PNG diagrams from Mermaid source files

# Source files
MERMAID_SOURCES := $(wildcard *.mmd)
# Target files
SVG_TARGETS := $(MERMAID_SOURCES:.mmd=.svg)
PNG_TARGETS := $(MERMAID_SOURCES:.mmd=.png)

.PHONY: all clean

all: $(SVG_TARGETS) $(PNG_TARGETS)

# Note: Using zsh -ic to ensure user's 'mmdc' alias is picked up from .zshrc
%.svg: %.mmd
	zsh -ic "mmdc -i $< -o $@"

%.png: %.mmd
	zsh -ic "mmdc -i $< -o $@ -b white -s 4"

clean:
	rm -f $(SVG_TARGETS) $(PNG_TARGETS)
