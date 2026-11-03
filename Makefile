preview:
	quarto preview
build:
	quarto render --to html
pdf:
	quarto render --to pdf
clean:
	rm -rf _site .quarto
