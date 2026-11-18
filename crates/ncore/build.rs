//! Compiles the CUDA kernels when nvcc is available (same detection as bsc-sys).
use std::path::PathBuf;

/// Same rules as bsc-sys: CUDA only when building for the host OS (a Windows GPU build is
/// made on Windows with Visual Studio + CUDA_PATH).
fn find_cuda(windows: bool, msvc: bool) -> Option<PathBuf> {
    if std::env::var_os("NZIP_NO_CUDA").is_some() || windows != cfg!(windows) || (windows && !msvc) {
        return None;
    }
    let mut c: Vec<PathBuf> = std::env::var_os("CUDA_PATH").map(PathBuf::from).into_iter().collect();
    if windows {
        return c.into_iter().find(|p| p.join("bin/nvcc.exe").exists() && p.join("lib/x64/cudart_static.lib").exists());
    }
    c.extend(["/opt/cuda", "/usr/local/cuda"].map(PathBuf::from));
    c.into_iter().find(|p| p.join("bin/nvcc").exists() && p.join("lib64/libcudart_static.a").exists())
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(ncore_cuda)");
    println!("cargo:rerun-if-changed=cuda/ncore.cu");
    println!("cargo:rerun-if-env-changed=NZIP_NO_CUDA");
    println!("cargo:rerun-if-env-changed=NZIP_PORTABLE");
    let portable = std::env::var("NZIP_PORTABLE").is_ok_and(|v| v == "1");
    let windows = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|v| v == "windows");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|v| v == "msvc");
    let Some(cuda) = find_cuda(windows, msvc) else { return };
    let mut b = cc::Build::new();
    b.cuda(true).cudart("static").compiler(cuda.join(if windows { "bin/nvcc.exe" } else { "bin/nvcc" }))
        .debug(false).no_default_flags(true).cpp_link_stdlib(None)
        .flag("-std=c++17").flag("-O3").flag("-lineinfo")
        .warnings(false);
    if windows {
        b.flag("-Xcompiler=/MD");
    } else {
        b.flag("-Xcompiler=-fPIC").flag("-ccbin=g++");
    }
    // Same GPU targets as bsc-sys (portable builds add datacenter archs + Turing PTX).
    let mut sms = vec!["75", "80", "86", "89", "120"];
    if portable {
        sms.extend(["90", "100"]);
        b.flag("-gencode=arch=compute_75,code=compute_75");
    }
    for sm in sms {
        b.flag(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
    }
    b.flag("-gencode=arch=compute_120,code=compute_120");
    b.file("cuda/ncore.cu");
    b.compile("ncore_cuda");
    println!("cargo:rustc-link-search=native={}", cuda.join(if windows { "lib/x64" } else { "lib64" }).display());
    println!("cargo:rustc-link-lib=static=cudadevrt");
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !windows {
        for l in ["dl", "rt", "pthread"] {
            println!("cargo:rustc-link-lib={l}");
        }
        // libstdc++ (needed by the CUDA host code): static for portable builds.
        if portable {
            let out = std::process::Command::new("g++").arg("-print-file-name=libstdc++.a").output().expect("g++");
            let path = std::path::PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
            println!("cargo:rustc-link-search=native={}", path.parent().expect("libstdc++.a").display());
            println!("cargo:rustc-link-lib=static=stdc++");
        } else {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
    println!("cargo:rustc-cfg=ncore_cuda");
}
