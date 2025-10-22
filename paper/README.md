# Paper Directory

LaTeX source files and compiled PDFs for the paper.

## Main Files

- **main.tex** - Main paper
- **main.pdf** - Compiled main paper
- **supplement.tex** - Supplemental material
- **supplement.pdf** - Compiled supplement
- **custom.bib** - Bibliography

## Figures

- **figs/source/** - Generated figures (PDFs from analysis)
  - Main figures: all_losses.pdf, stripplot.pdf, t_test.pdf, etc.
  - Variant figures: *_content.pdf, *_function.pdf, *_pos.pdf
  - Classification: classification_accuracy.pdf, wordcloud_*.pdf
- **figs/** - Additional figures and compiled multi-panel figures

## Compilation

Compile with standard LaTeX tools:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use your preferred LaTeX editor (Overleaf, TeXShop, etc.).

## Figure Generation

Figures are generated from model results:
```bash
# From repository root
./run_llm_stylometry.sh              # Generate all figures (all variants)
./run_llm_stylometry.sh -f 1a        # Generate specific figure
```

See main README for complete figure generation documentation.
