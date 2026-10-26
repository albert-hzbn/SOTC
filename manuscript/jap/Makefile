# Build the JAP submission. LaTeX auxiliary files go to build/; only the
# final PDFs are copied next to their sources. On Viper: module load texlive/2021

TEX  = pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build
DOCS = sotc_jap sotc_jap_supplementary cover_letter_jap
DEPS = refs_jap.bib $(wildcard tables/*.tex) $(wildcard figures/*.pdf)

all: $(DOCS:=.pdf)

%.pdf: %.tex $(DEPS)
	@mkdir -p build
	$(TEX) $* > /dev/null
	-cd build && BIBINPUTS=..: bibtex $* > /dev/null
	$(TEX) $* > /dev/null
	$(TEX) $* > /dev/null
	@cp build/$*.pdf $@
	@grep -c "Overfull\|undefined" build/$*.log | sed "s/^/$*: overfull or undefined: /"

clean:
	rm -rf build

.PHONY: all clean
