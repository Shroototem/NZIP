//! Builds the vendored libbsc as a static library.
//!
//! * CPU part: C/C++ with OpenMP, x86-64-v3 (AVX2).
//! * `NZIP_PORTABLE=1` (release builds for other machines): libstdc++ linked statically,
//!   extra GPU architectures (see `gpu_arch_flags`). The CPU target stays x86-64-v3: the
//!   runtime libraries this machine provides (CachyOS v3 packages) require AVX2 anyway.
//! * GPU part (libcubwt BWT + sort transform): compiled with nvcc when it is found
//!   (CUDA_PATH, /opt/cuda, /usr/local/cuda or PATH). The CUDA runtime is linked
//!   statically and loads the driver at run time, so the binary still runs, on CPU,
//!   on machines without an NVIDIA GPU or driver.
use std::path::{Path, PathBuf};

/// Target facts (the build script runs on the host; these describe what we build for).
pub struct Target {
    pub windows: bool,
    pub msvc: bool,
}
pub fn target() -> Target {
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    Target { windows: os == "windows", msvc: env == "msvc" }
}

/// CUDA toolkit for this target. nvcc can't cross-compile host code between OSes, so a
/// Windows GPU build must be made on Windows (Visual Studio + CUDA, found via CUDA_PATH).
fn find_cuda(t: &Target) -> Option<PathBuf> {
    if std::env::var_os("NZIP_NO_CUDA").is_some() {
        return None;
    }
    if t.windows != cfg!(windows) || (t.windows && !t.msvc) {
        return None;
    }
    let mut cands: Vec<PathBuf> = std::env::var_os("CUDA_PATH").map(PathBuf::from).into_iter().collect();
    if t.windows {
        return cands.into_iter().find(|p| p.join("bin/nvcc.exe").exists() && p.join("lib/x64/cudart_static.lib").exists());
    }
    cands.extend(["/opt/cuda", "/usr/local/cuda"].map(PathBuf::from));
    cands.into_iter().find(|p| p.join("bin/nvcc").exists() && p.join("lib64/libcudart_static.a").exists())
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/libbsc/libbsc");
    let cpp = [
        "adler32/adler32.cpp", "bwt/bwt.cpp", "coder/coder.cpp", "coder/qlfc/qlfc.cpp",
        "coder/qlfc/qlfc_model.cpp", "filters/detectors.cpp", "filters/preprocessing.cpp",
        "libbsc/libbsc.cpp", "lzp/lzp.cpp", "platform/platform.cpp", "st/st.cpp",
    ];
    let portable = portable();
    let t = target();
    let cuda = find_cuda(&t);
    println!("cargo:rustc-check-cfg=cfg(bsc_cuda)");
    println!("cargo:rerun-if-env-changed=NZIP_NO_CUDA");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NZIP_PORTABLE");

    // C/C++ flags per toolchain: GCC/Clang (Linux, llvm-mingw) or MSVC.
    let gnu_flags = |b: &mut cc::Build| {
        // Linux builds match this machine's x86-64-v3 runtime; Windows builds stay at v2
        // (any x86-64 CPU from the last ~15 years).
        b.flag("-fopenmp").flag(if t.windows { "-march=x86-64-v2" } else { "-march=x86-64-v3" });
    };
    let mut sais = cc::Build::new();
    sais.file(root.join("bwt/libsais/libsais.c")).define("LIBSAIS_OPENMP", None).opt_level(3).warnings(false);
    if t.msvc {
        sais.flag("/openmp:llvm");
    } else {
        if !t.windows {
            sais.compiler("gcc");
        }
        gnu_flags(&mut sais);
    }
    sais.compile("sais");

    let mut b = cc::Build::new();
    b.cpp(true).std("c++17").cpp_link_stdlib(None)
        .define("LIBBSC_OPENMP_SUPPORT", None)
        .define("LIBSAIS_OPENMP", None)
        .opt_level(3).warnings(false);
    if t.msvc {
        b.flag("/openmp:llvm");
    } else {
        if !t.windows {
            b.compiler("g++");
        }
        gnu_flags(&mut b);
    }
    if cuda.is_some() {
        b.define("LIBBSC_CUDA_SUPPORT", None);
    }
    for f in cpp {
        b.file(root.join(f));
    }
    b.compile("bsc");

    if let Some(cuda) = cuda {
        build_cuda(&root, &cuda, &t);
        println!("cargo:rustc-cfg=bsc_cuda");
    }
    if t.windows {
        // libbsc's Windows platform code asks for large-page privileges (advapi32).
        println!("cargo:rustc-link-lib=advapi32");
    }
    if t.msvc {
        // MSVC links its C++ runtime and LLVM OpenMP (libomp, shipped with Visual Studio) itself.
        println!("cargo:rustc-link-lib=libomp");
    } else if t.windows {
        // llvm-mingw: LLVM OpenMP (libomp.dll ships next to nzip.exe) and static libc++,
        // from the compiler's own library directory.
        let cxx = cc::Build::new().cpp(true).get_compiler();
        let out = std::process::Command::new(cxx.path()).arg("-print-file-name=libc++.a").output().expect("C++ compiler");
        let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if let Some(dir) = path.parent().filter(|d| d.is_absolute()) {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
        println!("cargo:rustc-link-lib=omp");
        println!("cargo:rustc-link-lib=static=c++");
    } else {
        println!("cargo:rustc-link-lib=gomp");
        link_stdcxx(portable);
    }
    println!("cargo:rerun-if-changed={}", root.display());
}

pub fn portable() -> bool {
    std::env::var("NZIP_PORTABLE").is_ok_and(|v| v == "1")
}

/// Link libstdc++: statically for portable builds (no dependency on the target's
/// libstdc++ version), dynamically otherwise.
fn link_stdcxx(portable: bool) {
    if portable {
        let out = std::process::Command::new("g++").arg("-print-file-name=libstdc++.a").output().expect("g++");
        let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        let dir = path.parent().expect("libstdc++.a location");
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static=stdc++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

/// GPU code: real code for Turing .. Blackwell consumer cards; portable builds add
/// datacenter Hopper/Blackwell and Turing-level PTX that any newer GPU can JIT-compile.
fn gpu_arch_flags(b: &mut cc::Build) {
    let mut sms = vec!["75", "80", "86", "89", "120"];
    if portable() {
        sms.extend(["90", "100"]);
        b.flag("-gencode=arch=compute_75,code=compute_75");
    }
    for sm in sms {
        b.flag(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
    }
    b.flag("-gencode=arch=compute_120,code=compute_120");
}

fn build_cuda(root: &Path, cuda: &Path, t: &Target) {
    let mut b = cc::Build::new();
    b.cuda(true).cudart("static").compiler(cuda.join(if t.windows { "bin/nvcc.exe" } else { "bin/nvcc" }))
        .debug(false).no_default_flags(true).cpp_link_stdlib(None)
        .define("LIBBSC_CUDA_SUPPORT", None)
        .define("LIBBSC_OPENMP_SUPPORT", None)
        .flag("-std=c++17").flag("-O3")
        .warnings(false);
    if t.windows {
        b.flag("-Xcompiler=/openmp:llvm,/MD");
    } else {
        b.flag("-Xcompiler=-fopenmp,-fPIC").flag("-ccbin=g++");
    }
    gpu_arch_flags(&mut b);
    b.file(root.join("bwt/libcubwt/libcubwt.cu")).file(root.join("st/st.cu"));
    b.compile("bsc_cuda");
    println!("cargo:rustc-link-search=native={}", cuda.join(if t.windows { "lib/x64" } else { "lib64" }).display());
    println!("cargo:rustc-link-lib=static=cudadevrt");
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !t.windows {
        for l in ["dl", "rt", "pthread"] {
            println!("cargo:rustc-link-lib={l}");
        }
    }
}
