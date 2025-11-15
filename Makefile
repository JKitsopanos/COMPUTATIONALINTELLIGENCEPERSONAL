clean:
	rm -f *.bib *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc

report.pdf:	report.tex
	pdflatex report
	biber report
	pdflatex report
	pdflatex report
