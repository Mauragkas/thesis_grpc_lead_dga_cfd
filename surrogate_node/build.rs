use std::env;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=native/incl");
    println!("cargo:rerun-if-changed=native/src");
    println!("cargo:rerun-if-changed=proto/surrogate.proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["proto/surrogate.proto"], &["proto"])?;

    // ── Base C++ build (CPU OpenMP + ROCm stubs) ─────────────────────── //
    let mut cpp_build = cc::Build::new();
    cpp_build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .include("native/incl")
        .include("native/incl/common")
        .include("native/incl/gp")
        .include("native/incl/knn")
        .include("native/incl/rf")
        .include("native/incl/mlp")
        // Common
        .file("native/src/common/gp_device.cpp")
        .file("native/src/common/gp_dispatcher.cpp")
        // Gaussian Process
        .file("native/src/gp/gp_cpu.cpp")
        .file("native/src/gp/gp_rocm.cpp")
        // k-Nearest Neighbours
        .file("native/src/knn/knn_cpu.cpp")
        .file("native/src/knn/knn_rocm.cpp")
        // Random Forest
        .file("native/src/rf/rf_cpu.cpp")
        .file("native/src/rf/rf_rocm.cpp")
        // Multi-Layer Perceptron (Neural Network)
        .file("native/src/mlp/mlp_cpu.cpp")
        .file("native/src/mlp/mlp_rocm.cpp")
        .flag_if_supported("-fopenmp")
        .flag_if_supported("/openmp");

    // ── CUDA support ─────────────────────────────────────────────────── //
    let nvcc_present = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or_else(|_| {
            Command::new("/opt/cuda/bin/nvcc")
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        });

    if nvcc_present {
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".to_string());

        let mut cuda_build = cc::Build::new();
        cuda_build
            .cuda(true)
            .include("native/incl")
            .include("native/incl/common")
            .include("native/incl/gp")
            .include("native/incl/knn")
            .include("native/incl/rf")
            .include("native/incl/mlp")
            .include(format!("{}/include", cuda_path))
            .file("native/src/gp/gp_cuda.cu")
            .file("native/src/knn/knn_cuda.cu")
            .file("native/src/rf/rf_cuda.cu")
            .file("native/src/mlp/mlp_cuda.cu")
            .flag("-O3")
            .flag("-DENABLE_CUDA");

        if cuda_build.try_compile("gp_cuda_native").is_ok() {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-lib=cudart");
            // Tell the C++ build that CUDA is available so stubs get elided.
            cpp_build.define("ENABLE_CUDA", "1");
        }
        // If try_compile fails we fall through: the CPU stubs already defined
        // knn_predict_cuda / rf_predict_cuda in the .cpp files.
        // We do NOT add .cu files to the C++ build — they contain CUDA syntax.
    }
    // When nvcc is absent: same outcome — CPU stubs provide all symbols.

    cpp_build.compile("gp_native");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=gomp");
    }

    Ok(())
}
