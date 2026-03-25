#!/bin/bash

# Compile the LLM Stylometry paper and supplement
#
# Usage:
#   ./paper/compile.sh           # Compile both main paper and supplement
#   ./paper/compile.sh main      # Compile main paper only
#   ./paper/compile.sh supplement # Compile supplement only
#   ./paper/compile.sh clean     # Remove build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

compile_tex() {
    local TEX_FILE=$1
    local NAME="${TEX_FILE%.tex}"

    echo -e "${BLUE}[INFO]${NC} Compiling ${TEX_FILE}..."

    # First pass
    pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1

    # BibTeX (only if .bib files are referenced)
    if grep -q '\\bibliography' "$TEX_FILE" 2>/dev/null; then
        echo -e "${BLUE}[INFO]${NC} Running BibTeX..."
        bibtex "$NAME" > /dev/null 2>&1 || true
        # Two more passes to resolve references
        pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1
    fi

    # Final pass
    pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1

    # Check for errors
    if [ -f "${NAME}.pdf" ]; then
        PAGES=$(pdfinfo "${NAME}.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "?")
        SIZE=$(ls -lh "${NAME}.pdf" | awk '{print $5}')
        echo -e "${GREEN}[OK]${NC} ${NAME}.pdf (${PAGES} pages, ${SIZE})"
    else
        echo -e "${RED}[ERROR]${NC} Failed to compile ${TEX_FILE}"
        # Show errors
        grep "^!" "${NAME}.log" 2>/dev/null || true
        exit 1
    fi

    # Check for undefined references
    UNDEF=$(grep -c "undefined" "${NAME}.log" 2>/dev/null | tail -1 || echo "0")
    if [ "$UNDEF" -gt 0 ]; then
        echo -e "${BLUE}[INFO]${NC} $UNDEF undefined reference warnings (check ${NAME}.log)"
    fi
}

clean() {
    echo -e "${BLUE}[INFO]${NC} Cleaning build artifacts..."
    rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz
    rm -f admin/*.aux admin/*.log admin/*.out
    echo -e "${GREEN}[OK]${NC} Clean complete"
}

case "${1:-all}" in
    main)
        compile_tex main.tex
        ;;
    supplement)
        compile_tex supplement.tex
        ;;
    response)
        cd admin
        compile_tex response_letter.tex
        ;;
    clean)
        clean
        ;;
    all)
        compile_tex main.tex
        compile_tex supplement.tex
        cd admin
        compile_tex response_letter.tex
        ;;
    *)
        echo "Usage: $0 [main|supplement|response|clean|all]"
        exit 1
        ;;
esac
