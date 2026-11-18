//! GPU encoder/decoder (CUDA). Produces streams byte-identical to [`crate::cpu`].

use crate::stream::{self, Layout};
use crate::{Error, FRAME};

// The GPU writes and reads the stream's little-endian u16 words directly in host memory.
#[cfg(all(ncore_cuda, target_endian = "big"))]
compile_error!("the ncore GPU path assumes a little-endian host");

#[cfg(ncore_cuda)]
unsafe extern "C" {
    fn ncore_gpu_available() -> i32;
    fn ncore_gpu_warmup() -> i32;
    fn ncore_gpu_encode(data: *const u8, total: u64, lane_len: u32, cap: u64, counts: *mut u32, packed: *mut u16, packed_cap: u64,
                        packed_words: *mut u64, kernel_ms: *mut f32) -> i32;
    fn ncore_gpu_decode(words: *const u8, n_words: u64, offs: *const u64, out: *mut u8, total: u64, lane_len: u32, status: *mut i32,
                        kernel_ms: *mut f32) -> i32;
}

/// True if the kernels were built and a GPU is usable.
pub fn available() -> bool {
    #[cfg(ncore_cuda)]
    {
        static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *A.get_or_init(|| unsafe { ncore_gpu_available() != 0 })
    }
    #[cfg(not(ncore_cuda))]
    false
}

/// Cheap check, without touching the CUDA runtime: kernels were built and an NVIDIA
/// driver device node exists. Initialising CUDA costs ~0.1-0.2 s, so callers use this to
/// decide whether starting a background [`warmup`] is worth it.
pub fn likely_present() -> bool {
    cfg!(ncore_cuda) && nvidia_driver_present()
}

/// An NVIDIA driver is installed: Linux has its device node, Windows its user-mode driver.
fn nvidia_driver_present() -> bool {
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
        std::path::Path::new(&root).join("System32").join("nvcuda.dll").exists()
    } else {
        std::path::Path::new("/dev/nvidiactl").exists()
    }
}

/// Pay the one-time CUDA start-up (context creation and kernel module load) now,
/// typically on a background thread while the CPU does other work. Returns whether the
/// GPU is usable. Safe to call more than once and from several threads.
pub fn warmup() -> bool {
    #[cfg(ncore_cuda)]
    {
        static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *W.get_or_init(|| available() && unsafe { ncore_gpu_warmup() } == 0)
    }
    #[cfg(not(ncore_cuda))]
    false
}

/// Max words a lane of `lane_len` bytes can produce (1 per symbol + 2 per frame flush).
fn lane_cap(lane_len: usize) -> usize {
    lane_len + 2 * lane_len.div_ceil(FRAME) + 2
}

/// Encode on the GPU. Returns (stream, kernel milliseconds).
///
/// The kernels pack every lane's words on the GPU and copy them straight into the
/// output after its header, so the stream is assembled with a single device-to-host copy.
pub fn encode(data: &[u8], lane_len: usize) -> Result<(Vec<u8>, f32), Error> {
    if !available() {
        return Err(Error::Gpu("no usable GPU".into()));
    }
    let layout = Layout::new(data.len(), lane_len);
    if layout.n_lanes == 0 {
        return Ok((stream::write(&layout, &[]), 0.0));
    }
    let cap = lane_cap(layout.lane_len);
    let hdr = stream::header_len(layout.n_lanes);
    // Worst-case bound; only the pages actually written get touched.
    let max_words = layout.n_lanes * cap;
    let mut out: Vec<u8> = Vec::with_capacity(hdr + 2 * max_words);
    let mut counts = vec![0u32; layout.n_lanes];
    let mut words = 0u64;
    #[cfg_attr(not(ncore_cuda), allow(unused_mut))]
    let mut ms = 0f32;
    #[cfg(ncore_cuda)]
    let rc = unsafe {
        // Packed words start after the header (16 + 4 * n_lanes bytes: 2-byte aligned).
        let packed = out.as_mut_ptr().add(hdr).cast::<u16>();
        ncore_gpu_encode(data.as_ptr(), data.len() as u64, layout.lane_len as u32, cap as u64, counts.as_mut_ptr(), packed,
                         max_words as u64, &mut words, &mut ms)
    };
    #[cfg(not(ncore_cuda))]
    let rc = { let _ = (&mut counts, &mut words, cap, max_words); -1 };
    if rc != 0 {
        return Err(Error::Gpu(format!("encode failed (CUDA error {rc})")));
    }
    // SAFETY: the GPU initialised bytes hdr..hdr + 2 * words; the header is written next.
    unsafe { out.set_len(hdr + 2 * words as usize) };
    let mut head = Vec::with_capacity(hdr);
    stream::write_header(&layout, counts.iter().copied(), &mut head);
    out[..hdr].copy_from_slice(&head);
    Ok((out, ms))
}

/// Decode on the GPU. Returns (data, kernel milliseconds).
pub fn decode(b: &[u8]) -> Result<(Vec<u8>, f32), Error> {
    let (layout, _, _) = stream::parse(b)?;
    let mut out = vec![0u8; layout.total];
    let ms = decode_into(b, &mut out)?;
    Ok((out, ms))
}

/// Decode on the GPU straight into `out` (exactly the original length). Returns kernel
/// milliseconds. The stream's word area is copied to the GPU as is, with no host-side
/// conversion.
pub fn decode_into(b: &[u8], out: &mut [u8]) -> Result<f32, Error> {
    if !available() {
        return Err(Error::Gpu("no usable GPU".into()));
    }
    let (layout, offs, body) = stream::parse(b)?;
    if out.len() != layout.total {
        return Err(Error::Corrupt("output length mismatch"));
    }
    if layout.n_lanes == 0 {
        return Ok(0.0);
    }
    let offs64: Vec<u64> = offs.iter().map(|&o| o as u64).collect();
    let mut status = vec![0i32; layout.n_lanes];
    #[cfg_attr(not(ncore_cuda), allow(unused_mut))]
    let mut ms = 0f32;
    #[cfg(ncore_cuda)]
    let rc = unsafe {
        ncore_gpu_decode(body.as_ptr(), (body.len() / 2) as u64, offs64.as_ptr(), out.as_mut_ptr(), layout.total as u64,
                         layout.lane_len as u32, status.as_mut_ptr(), &mut ms)
    };
    #[cfg(not(ncore_cuda))]
    let rc = { let _ = (&offs64, body, &mut status); -1 };
    if rc != 0 {
        return Err(Error::Gpu(format!("decode failed (CUDA error {rc})")));
    }
    if status.iter().any(|&s| s != 0) {
        return Err(Error::Corrupt("lane decode error"));
    }
    Ok(ms)
}
