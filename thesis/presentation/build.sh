#!/usr/bin/env bash
# ==============================================================================
# Presentation Build Script (XeLaTeX)
# Τμήμα Μηχανικών Η/Υ & Πληροφορικής (CEID), Πανεπιστήμιο Πατρών
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

OUTPUT_PDF="presentation.pdf"
BUILD_DIR="build"
MAIN_TEX="main.tex"

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_step()    { echo -e "${BLUE}${BOLD}[>>]${NC} ${BOLD}$1${NC}"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }

clean() {
    log_info "Καθαρισμός βοηθητικών αρχείων και φακέλου build/..."
    rm -rf "${BUILD_DIR}"
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb "${OUTPUT_PDF}" main.pdf
    log_success "Καθαρίστηκαν πλήρως."
    exit 0
}

if [[ "${1:-}" == "--clean" ]]; then
    clean
fi

mkdir -p "${BUILD_DIR}"

QUICK=false
if [[ "${1:-}" == "-q" || "${1:-}" == "--quick" ]]; then
    QUICK=true
fi

log_step "Μεταγλώττιση Παρουσίασης XeLaTeX (Pass 1)..."
xelatex -interaction=nonstopmode -output-directory="${BUILD_DIR}" "${MAIN_TEX}" > "${BUILD_DIR}/xelatex_pass1.log" 2>&1 || {
    echo -e "${YELLOW}--- Σφάλματα από το log ---${NC}"
    grep -E -A 2 "^! " "${BUILD_DIR}/main.log" 2>/dev/null | head -n 30 || cat "${BUILD_DIR}/xelatex_pass1.log" | tail -n 25
    exit 1
}

if [ "${QUICK}" = false ]; then
    log_step "Μεταγλώττιση Παρουσίασης XeLaTeX (Pass 2)..."
    xelatex -interaction=nonstopmode -output-directory="${BUILD_DIR}" "${MAIN_TEX}" > "${BUILD_DIR}/xelatex_pass2.log" 2>&1 || {
        echo -e "${YELLOW}--- Σφάλματα από το log ---${NC}"
        grep -E -A 2 "^! " "${BUILD_DIR}/main.log" 2>/dev/null | head -n 30 || cat "${BUILD_DIR}/xelatex_pass2.log" | tail -n 25
        exit 1
    }
fi

if [ -f "${BUILD_DIR}/main.pdf" ]; then
    cp "${BUILD_DIR}/main.pdf" "${OUTPUT_PDF}"
    PAGES=$(pdfinfo "${OUTPUT_PDF}" 2>/dev/null | grep -i "^Pages:" | awk '{print $2}' || echo "N/A")
    SIZE=$(du -h "${OUTPUT_PDF}" | awk '{print $1}')
    echo ""
    log_success "Η παρουσίαση παρήχθη επιτυχώς: ${OUTPUT_PDF}"
    echo -e "  ${BOLD}Σελίδες:${NC} ${PAGES}"
    echo -e "  ${BOLD}Μέγεθος:${NC} ${SIZE}"
    echo ""
else
    echo -e "${YELLOW}[!] Δεν βρέθηκε το παραγόμενο PDF στο ${BUILD_DIR}/main.pdf${NC}"
    exit 1
fi
