#!/usr/bin/env bash
# ==============================================================================
# AMD ROCm / HIP GPU Benchmark Runner & Plotter
# Runs GP covariance & MLP inference benchmarks on AMD GPU vs OpenMP CPU,
# generates updated figures, and optionally recompiles thesis and presentation.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_step()    { echo -e "${BLUE}${BOLD}[>>]${NC} ${BOLD}$1${NC}"; }
log_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "\033[0;31m[ERROR]${NC} $1" >&2; }

REBUILD_DOCS=false
for arg in "$@"; do
    case "${arg}" in
        --rebuild-docs|-r)
            REBUILD_DOCS=true
            ;;
        --help|-h)
            echo "Usage: ./run_amd_gpu_benchmarks.sh [--rebuild-docs]"
            echo "  --rebuild-docs, -r   Recompile thesis and presentation after generating new figures"
            exit 0
            ;;
    esac
done

# 1. Locate hipcc
HIPCC=""
if command -v hipcc &> /dev/null; then
    HIPCC="hipcc"
elif [ -x "/opt/rocm/bin/hipcc" ]; then
    HIPCC="/opt/rocm/bin/hipcc"
elif compgen -G "/opt/rocm-*/bin/hipcc" > /dev/null; then
    HIPCC=$(ls -d /opt/rocm-*/bin/hipcc | tail -n 1)
fi

if [ -z "${HIPCC}" ]; then
    log_error "hipcc compiler not found in PATH or /opt/rocm/bin. Please install or expose ROCm."
    exit 1
fi

log_info "Using HIP compiler: ${HIPCC}"

# 2. Compile benchmark binary
CPP_SRC="evaluation/scripts/bench_amd_gpu.cpp"
BIN_OUT="evaluation/scripts/bench_amd_gpu_bin"

log_step "Compiling AMD GPU benchmark (${CPP_SRC})..."
${HIPCC} -O3 -fopenmp "${CPP_SRC}" -o "${BIN_OUT}"
log_success "Compilation successful: ${BIN_OUT}"

# 3. Run benchmark
JSON_OUT="evaluation/data/amd_gpu_benchmarks.json"
mkdir -p "evaluation/data" "evaluation/figures"

log_step "Executing 10-trial statistical benchmarks on AMD ROCm device and Host CPU..."
"${BIN_OUT}" "${JSON_OUT}" 10

# 4. Generate updated publication figure
log_step "Generating publication figure (fig2_surrogate_cuda_vs_cpu.png)..."
python3 evaluation/scripts/plot_surrogate_gpu_benchmarks.py --out-dir evaluation/figures
log_success "Generated evaluation/figures/fig2_surrogate_cuda_vs_cpu.png"

# Clean up binary
rm -f "${BIN_OUT}"

# 5. Optionally Rebuild Thesis and Presentation
if [ "${REBUILD_DOCS}" = true ]; then
    echo ""
    log_step "Rebuilding Thesis PDF (XeLaTeX + BibTeX)..."
    (
        cd thesis
        ./build.sh --quick
    )
    log_success "Thesis PDF updated."

    echo ""
    log_step "Rebuilding Presentation PDF (XeLaTeX)..."
    (
        cd thesis/presentation
        ./build.sh
    )
    log_success "Presentation PDF updated."
fi

echo ""
log_success "All tasks completed successfully!"
