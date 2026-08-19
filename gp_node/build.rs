use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=native/incl");
    println!("cargo:rerun-if-changed=native/src");

    let mut cpp_build = cc::Build::new();
    cpp_build
        .cpp(true)
        .std("c++17")
        .opt_level(3)
        .include("native/incl")
        .file("native/src/gp_cpu.cpp")
        .file("native/src/gp_device.cpp")
        .file("native/src/gp_dispatcher.cpp")
        .file("native/src/gp_rocm.cpp")
        .flag_if_supported("-fopenmp")
        .flag_if_supported("/openmp");

    // Check if CUDA nvcc compiler exists
    let has_cuda = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or_else(|_| {
            Command::new("/opt/cuda/bin/nvcc")
                .arg("--version")
                .output()
                .map(|out| out.status.success())
                .unwrap_or(false)
        });

    if has_cuda {
        let cuda_path = env::var("CUDA_PATH")
            .unwrap_or_else(|_| "/opt/cuda".to_string());
        
        let mut cuda_build = cc::Build::new();
        cuda_build
            .cuda(true)
            .include("native/incl")
            .include(format!("{}/include", cuda_path))
            .file("native/src/gp_cuda.cu")
            .flag("-O3")
            .flag("-DENABLE_CUDA");

        if cuda_build.try_compile("gp_cuda_native").is_ok() {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-lib=cudart");
            cpp_build.define("ENABLE_CUDA", "1");
        } else {
            cpp_build.file("native/src/gp_cuda.cu");
        }
    } else {
        cpp_build.file("native/src/gp_cuda.cu");
    }

    cpp_build.compile("gp_native");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=gomp");
    }
}
