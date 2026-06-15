TEX = main.tex
PDF = main.pdf
CACHE = ../.tectonic-cache

.PHONY: all clean

all:
	XDG_CACHE_HOME=$(abspath $(CACHE)) tectonic -X compile $(TEX)

xelatex:
	xelatex -interaction=nonstopmode $(TEX)
	xelatex -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
