#!/usr/bin/env bash
set -euo pipefail

INPUT_MD="thesis_progress_report_el.md"
OUTPUT_PDF="thesis_progress_report_el.pdf"

echo ">> Generating ${OUTPUT_PDF} from ${INPUT_MD} using Pandoc & XeLaTeX..."

pandoc "${INPUT_MD}" -o "${OUTPUT_PDF}" \
  --pdf-engine=xelatex \
  --toc \
  -V geometry:"left=0.6in,right=0.6in,top=0.7in,bottom=0.7in" \
  -V mainfont="Linux Libertine O" \
  -V monofont="DejaVu Sans Mono" \
  -V lang=el

echo ">> Successfully generated: ${OUTPUT_PDF}"
