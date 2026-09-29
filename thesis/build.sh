#!/usr/bin/env bash
# ==============================================================================
# Thesis Build & Concatenation Script (XeLaTeX + BibTeX)
# Τμήμα Μηχανικών Η/Υ & Πληροφορικής (CEID), Πανεπιστήμιο Πατρών
# ==============================================================================
set -euo pipefail

# Μετακίνηση στον κατάλογο του script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Χρώματα κονσόλας
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

OUTPUT_PDF="thesis.pdf"
BUILD_DIR="build"
MAIN_TEX="main.tex"
CHAPTERS_MANIFEST="_all_chapters.tex"
APPENDICES_MANIFEST="_all_appendices.tex"

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_step()    { echo -e "${BLUE}${BOLD}[>>]${NC} ${BOLD}$1${NC}"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}${BOLD}[ERROR]${NC} $1" >&2; }

usage() {
    cat << EOF
Χρήση: ./build.sh [ΕΠΙΛΟΓΕΣ]

Επιλογές:
  (καμία επιλογή)  Πλήρης μεταγλώττιση (XeLaTeX -> BibTeX -> XeLaTeX x2)
  -c, --concat     Συνένωση (concatenation) όλων των .tex αρχείων σε ένα ενιαίο αρχείο
                   (build/thesis_concatenated.tex) και μεταγλώττιση.
  -q, --quick      Γρήγορη μεταγλώττιση 1 περάσματος (single-pass XeLaTeX) για γρήγορο preview.
  --clean          Διαγραφή όλων των ενδιάμεσων αρχείων μεταγλώττισης (build/).
  -w, --watch      Παρακολούθηση αρχείων (.tex, .bib) και αυτόματη επαναμεταγλώττιση κατά την αποθήκευση.
  -h, --help       Εμφάνιση αυτού του μηνύματος βοήθειας.

Παραδείγματα:
  ./build.sh               # Κανονική πλήρης παραγωγή του thesis.pdf
  ./build.sh --quick       # Γρήγορη ενημέρωση του PDF ενώ γράφετε
  ./build.sh --concat      # Παραγωγή ενιαίου .tex αρχείου και compilation
  ./build.sh --clean       # Καθαρισμός φακέλου build
EOF
    exit 0
}

clean_build() {
    log_info "Καθαρισμός ενδιάμεσων αρχείων μεταγλώττισης..."
    rm -rf "${BUILD_DIR}"
    rm -f "${CHAPTERS_MANIFEST}" "${APPENDICES_MANIFEST}" "${OUTPUT_PDF}"
    rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot *.loa *.fls *.fdb_latexmk
    log_success "Ο κατάλογος καθαρίστηκε πλήρως."
    exit 0
}

# ------------------------------------------------------------------------------
# 1. Δυναμική Ανακάλυψη & Ταξινόμηση Κεφαλαίων σε Σειρά
# ------------------------------------------------------------------------------
generate_manifests() {
    log_info "Σάρωση και ταξινόμηση αρχείων κεφαλαίων και παραρτημάτων..."
    
    # 1. Παραγωγή _all_chapters.tex
    echo "% Αυτόματα παραγόμενος κατάλογος κεφαλαίων κατά αύξουσα σειρά" > "${CHAPTERS_MANIFEST}"
    echo "% ΜΗΝ ΤΟ ΕΠΕΞΕΡΓΑΖΕΣΤΕ ΧΕΙΡΟΚΙΝΗΤΑ - Παράγεται από το build.sh" >> "${CHAPTERS_MANIFEST}"
    echo "" >> "${CHAPTERS_MANIFEST}"

    local chapter_count=0
    local section_count=0

    # Εύρεση όλων των καταλόγων κεφαλαίων ταξινομημένων
    while IFS= read -r ch_dir; do
        if [ -d "${ch_dir}" ]; then
            ch_name=$(basename "${ch_dir}")
            echo "% --- ${ch_name} ---" >> "${CHAPTERS_MANIFEST}"
            
            # Εύρεση όλων των .tex αρχείων μέσα στον κατάλογο (συμπεριλαμβανομένων υποφακέλων)
            while IFS= read -r tex_file; do
                if [ -f "${tex_file}" ]; then
                    rel_path="${tex_file#./}"
                    echo "\\input{${rel_path}}" >> "${CHAPTERS_MANIFEST}"
                    section_count=$((section_count + 1))
                fi
            done < <(find "${ch_dir}" -type f -name "*.tex" | sort)
            
            echo "" >> "${CHAPTERS_MANIFEST}"
            chapter_count=$((chapter_count + 1))
        fi
    done < <(find chapters -mindepth 1 -maxdepth 1 -type d | sort)

    log_info "Βρέθηκαν ${chapter_count} κεφάλαια (${section_count} συνολικά .tex αρχεία)."

    # 2. Παραγωγή _all_appendices.tex
    echo "% Αυτόματα παραγόμενος κατάλογος παραρτημάτων κατά αύξουσα σειρά" > "${APPENDICES_MANIFEST}"
    if [ -d "appendices" ]; then
        while IFS= read -r app_file; do
            if [ -f "${app_file}" ]; then
                rel_path="${app_file#./}"
                echo "\\input{${rel_path}}" >> "${APPENDICES_MANIFEST}"
            fi
        done < <(find appendices -type f -name "*.tex" | sort)
    fi
}

# ------------------------------------------------------------------------------
# 2. Συνένωση (Concatenation) όλων των αρχείων σε ένα αυτόνομο .tex
# ------------------------------------------------------------------------------
concatenate_to_single_file() {
    local target_file="${BUILD_DIR}/thesis_concatenated.tex"
    mkdir -p "${BUILD_DIR}"
    log_step "Συνένωση όλων των .tex αρχείων σε ενιαίο αρχείο: ${target_file}"

    {
        echo "% =============================================================================="
        echo "% ΕΝΙΑΙΟ ΣΥΝΕΝΩΜΕΝΟ ΑΡΧΕΙΟ ΔΙΠΛΩΜΑΤΙΚΗΣ ΕΡΓΑΣΙΑΣ (STANDALONE CONCATENATED TEX)"
        echo "% Δημιουργήθηκε αυτόματα: $(date)"
        echo "% =============================================================================="
        echo ""
        echo "\\documentclass[11pt,a4paper,twoside,openright]{report}"
        echo ""
        
        # Ενσωμάτωση μεταδεδομένων και προοιμίου
        echo "% --- Μεταδεδομένα (metadata.tex) ---"
        cat metadata.tex
        echo ""
        echo "% --- Προοίμιο (preamble.tex) ---"
        cat preamble.tex
        echo ""
        echo "\\begin{document}"
        echo "\\pagenumbering{roman}"
        echo ""
        
        # Frontmatter
        echo "% =============================================================================="
        echo "% ΠΡΟΚΑΤΑΡΚΤΙΚΕΣ ΣΕΛΙΔΕΣ (Frontmatter)"
        echo "% =============================================================================="
        while IFS= read -r fm_file; do
            echo "% --- File: ${fm_file} ---"
            cat "${fm_file}"
            echo ""
            echo "\\clearemptydoublepage"
        done < <(find frontmatter -type f -name "*.tex" | sort)

        # Πίνακες Περιεχομένων
        echo "\\tableofcontents"
        echo "\\clearemptydoublepage"
        echo "\\listoffigures"
        echo "\\addcontentsline{toc}{chapter}{\\listfigurename}"
        echo "\\clearemptydoublepage"
        echo "\\listoftables"
        echo "\\addcontentsline{toc}{chapter}{\\listtablename}"
        echo "\\clearemptydoublepage"

        # Mainmatter
        echo "% =============================================================================="
        echo "% ΚΥΡΙΩΣ ΣΩΜΑ ΕΡΓΑΣΙΑΣ (Mainmatter)"
        echo "% =============================================================================="
        echo "\\clearemptydoublepage"
        echo "\\pagenumbering{arabic}"
        echo "\\setcounter{page}{1}"
        echo ""

        while IFS= read -r ch_dir; do
            ch_name=$(basename "${ch_dir}")
            echo "% =========================================================================="
            echo "% ΚΕΦΑΛΑΙΟ: ${ch_name}"
            echo "% =========================================================================="
            while IFS= read -r tex_file; do
                echo "% --- Snippet: ${tex_file} ---"
                cat "${tex_file}"
                echo ""
            done < <(find "${ch_dir}" -type f -name "*.tex" | sort)
        done < <(find chapters -mindepth 1 -maxdepth 1 -type d | sort)

        # Βιβλιογραφία
        echo "% =============================================================================="
        echo "% ΒΙΒΛΙΟΓΡΑΦΙΑ"
        echo "% =============================================================================="
        echo "\\clearemptydoublepage"
        echo "\\bibliographystyle{ieeetr}"
        echo "\\addcontentsline{toc}{chapter}{\\bibname}"
        echo "\\bibliography{bibliography/references}"
        echo ""

        # Παραρτήματα
        echo "% =============================================================================="
        echo "% ΠΑΡΑΡΤΗΜΑΤΑ"
        echo "% =============================================================================="
        echo "\\clearemptydoublepage"
        echo "\\appendix"
        while IFS= read -r app_file; do
            echo "% --- Snippet: ${app_file} ---"
            cat "${app_file}"
            echo ""
        done < <(find appendices -type f -name "*.tex" | sort)

        echo "\\end{document}"
    } > "${target_file}"

    log_success "Το ενιαίο αρχείο δημιουργήθηκε επιτυχώς (${target_file})."
}

# ------------------------------------------------------------------------------
# 3. Συγχρονισμός & Σύνδεση Σχημάτων Αξιολόγησης (evaluation/figures)
# ------------------------------------------------------------------------------
sync_evaluation_figures() {
    log_info "Συγχρονισμός και σύνδεση σχημάτων και πινάκων αξιολόγησης (evaluation/figures)..."
    mkdir -p figures
    if [ -d "../evaluation/figures" ]; then
        for f in ../evaluation/figures/*; do
            if [ -e "$f" ]; then
                fname=$(basename "$f")
                target="figures/${fname}"
                if [ ! -e "$target" ] && [ ! -L "$target" ]; then
                    ln -sf "../../evaluation/figures/${fname}" "$target"
                fi
            fi
        done
    fi
}

# ------------------------------------------------------------------------------
# 4. Μεταγλώττιση (Compilation Pipeline)
# ------------------------------------------------------------------------------
run_xelatex() {
    local target="$1"
    local pass_label="$2"
    local target_base
    target_base="$(basename "${target%.tex}")"
    local log_file="${BUILD_DIR}/${target_base}.log"

    log_info "Εκτέλεση XeLaTeX [${pass_label}]..."
    if ! xelatex -interaction=nonstopmode -output-directory="${BUILD_DIR}" "${target}" > "${BUILD_DIR}/xelatex.stdout.log" 2>&1; then
        log_error "Αποτυχία κατά την εκτέλεση XeLaTeX!"
        echo -e "${YELLOW}--- Σφάλματα από το log (${log_file}) ---${NC}"
        grep -E -A 2 "^! " "${log_file}" 2>/dev/null | head -n 30 || cat "${BUILD_DIR}/xelatex.stdout.log" | tail -n 25
        exit 1
    fi
}

run_bibtex() {
    local aux_name="$1"
    log_info "Εκτέλεση BibTeX για παραπομπές..."
    (
        cd "${BUILD_DIR}"
        if ! BIBINPUTS="..:../bibliography:${BIBINPUTS:-}" BSTINPUTS="..:${BSTINPUTS:-}" bibtex "${aux_name}" > bibtex.stdout.log 2>&1; then
            log_warn "Το BibTeX εμφάνισε προειδοποιήσεις (ελέγξτε ${BUILD_DIR}/bibtex.stdout.log):"
            tail -n 10 bibtex.stdout.log || true
        fi
    )
}

compile_document() {
    local mode="${1:-full}"
    local use_concat="${2:-false}"
    local target_tex="${MAIN_TEX}"
    local base_name="main"

    mkdir -p "${BUILD_DIR}"
    sync_evaluation_figures

    if [ "${use_concat}" = "true" ]; then
        concatenate_to_single_file
        target_tex="${BUILD_DIR}/thesis_concatenated.tex"
        base_name="thesis_concatenated"
    else
        generate_manifests
    fi

    log_step "Έναρξη διαδικασίας παραγωγής PDF..."

    if [ "${mode}" = "quick" ]; then
        run_xelatex "${target_tex}" "Γρήγορο πέρασμα 1/1"
    else
        run_xelatex "${target_tex}" "Πέρασμα 1/3 (Δημιουργία δεικτών)"
        run_bibtex "${base_name}"
        run_xelatex "${target_tex}" "Πέρασμα 2/3 (Ενσωμάτωση βιβλιογραφίας & TOC)"
        run_xelatex "${target_tex}" "Πέρασμα 3/3 (Επίλυση διασταυρούμενων αναφορών)"
    fi

    # Αντιγραφή του παραγόμενου PDF στον ριζικό φάκελο thesis/
    if [ -f "${BUILD_DIR}/${base_name}.pdf" ]; then
        cp "${BUILD_DIR}/${base_name}.pdf" "${OUTPUT_PDF}"
        local page_count
        page_count=$(pdfinfo "${OUTPUT_PDF}" 2>/dev/null | grep -i "^Pages:" | awk '{print $2}' || echo "N/A")
        local file_size
        file_size=$(du -h "${OUTPUT_PDF}" | awk '{print $1}')
        
        echo ""
        log_success "Η παραγωγή ολοκληρώθηκε επιτυχώς!"
        echo -e "  ${BOLD}Αρχείο εξόδου:${NC} ${GREEN}${SCRIPT_DIR}/${OUTPUT_PDF}${NC}"
        echo -e "  ${BOLD}Σελίδες:${NC}       ${page_count}"
        echo -e "  ${BOLD}Μέγεθος:${NC}       ${file_size}"
        echo ""
    else
        log_error "Δεν βρέθηκε παραγόμενο PDF στο ${BUILD_DIR}/${base_name}.pdf"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# 5. Λειτουργία Παρακολούθησης (Watch Mode)
# ------------------------------------------------------------------------------
watch_mode() {
    log_step "Ενεργοποίηση λειτουργίας παρακολούθησης (Watch Mode)..."
    log_info "Παρακολουθούνται αλλαγές σε αρχεία .tex και .bib..."
    log_info "Πιέστε Ctrl+C για έξοδο."
    
    # Πρώτη μεταγλώττιση
    compile_document "quick" "false"

    if which inotifywait >/dev/null 2>&1; then
        while true; do
            inotifywait -q -r -e modify,create,delete --exclude "build/.*" .
            echo ""
            log_info "Εντοπίστηκε αλλαγή! Επαναμεταγλώττιση..."
            compile_document "quick" "false"
        done
    else
        log_warn "Το 'inotifywait' δεν είναι εγκατεστημένο. Χρήση polling loop 2 δευτερολέπτων..."
        local last_checksum=""
        while true; do
            current_checksum=$(find . -name "*.tex" -o -name "*.bib" | grep -v "/build/" | xargs stat -c "%Y %n" 2>/dev/null | md5sum)
            if [ "${last_checksum}" != "" ] && [ "${last_checksum}" != "${current_checksum}" ]; then
                echo ""
                log_info "Εντοπίστηκε αλλαγή! Επαναμεταγλώττιση..."
                compile_document "quick" "false"
            fi
            last_checksum="${current_checksum}"
            sleep 2
        done
    fi
}

# ------------------------------------------------------------------------------
# Επεξεργασία Ορισμάτων Γραμμής Εντολών
# ------------------------------------------------------------------------------
MODE="full"
USE_CONCAT="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --clean)
            clean_build
            ;;
        -q|--quick)
            MODE="quick"
            shift
            ;;
        -c|--concat|--single-file)
            USE_CONCAT="true"
            shift
            ;;
        -w|--watch)
            watch_mode
            ;;
        *)
            log_error "Άγνωστη επιλογή: $1"
            usage
            ;;
    esac
done

compile_document "${MODE}" "${USE_CONCAT}"
