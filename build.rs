//! Embeds the GPU kernels (gpu/sa.ptx, ~260 KB of text) as a ~50 KB raw deflate stream,
//! written with the crate's own zlib port; gpu.rs inflates it when the GPU starts.

#[allow(dead_code)]
#[path = "src/zlib.rs"]
mod zlib;

fn main() {
    println!("cargo::rerun-if-changed=gpu/sa.ptx");
    println!("cargo::rerun-if-changed=src/zlib.rs");
    let ptx = std::fs::read("gpu/sa.ptx").expect("read gpu/sa.ptx");
    let packed = zlib::deflate(&ptx, 9, 9);
    let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("sa.ptx.z");
    std::fs::write(out, packed).expect("write sa.ptx.z");
}
