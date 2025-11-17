clean:
	rm -f refs.bib *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc report.pdf

report.pdf:	report.tex
	pdflatex report
	biber report
	pdflatex report
	pdflatex report
