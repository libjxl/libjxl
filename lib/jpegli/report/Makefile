all: jpegli.pdf

#
# To build:
# env JHEP_REPORTS_CSL=[path to jhep-reports.csl] PANDOC_EQNOS_BINARY=[path pandoc-eqnos] make
#
# jhep-reports.csl can be downloaded from https://paperpile.com/s/jhep-reports-citation-style/
#
# pandoc-eqnos can be downloaded from https://github.com/tomduck/pandoc-eqnos
#

jpegli.pdf: jpegli.md meta.yaml bibliography.bib
	pandoc -f markdown -t pdf --citeproc --bibliography bibliography.bib --csl ${JHEP_REPORTS_CSL} --filter ${PANDOC_EQNOS_BINARY} --metadata-file meta.yaml --template pandoc_template.tex jpegli.md > jpegli.pdf

show: jpegli.pdf
	xdg-open jpegli.pdf
