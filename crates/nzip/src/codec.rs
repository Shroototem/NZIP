//! Block codecs: store, zstd, bsc. Each block is compressed independently.

use crate::format::{CODEC_BSC, CODEC_NCORE, CODEC_STORE, CODEC_ZSTD};

/// Lane size for the ncore codec (measured sweet spot: 64 KiB).
pub const NCORE_LANE: usize = 64 * 1024;

/// Serialises GPU calls so concurrent blocks don't each allocate a full set of buffers.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Store,
    Zstd { level: i32 },
    /// `gpu`: run the BWT on the GPU (byte-identical output; CPU fallback inside libbsc).
    Bsc { coder: i32, gpu: bool },
    /// GPU entropy core with its order-2 test model (experimental).
    Ncore { gpu: bool },
}

impl Codec {
    pub fn id(&self) -> u8 {
        match self {
            Codec::Store => CODEC_STORE,
            Codec::Zstd { .. } => CODEC_ZSTD,
            Codec::Bsc { .. } => CODEC_BSC,
            Codec::Ncore { .. } => CODEC_NCORE,
        }
    }

    /// `threads` = threads this single call may use internally (bsc only).
    pub fn compress(&self, data: &[u8], threads: usize) -> Result<Vec<u8>, Error> {
        match *self {
            Codec::Store => Ok(data.to_vec()),
            Codec::Zstd { level } => zstd::bulk::compress(data, level).map_err(|e| Error::Codec(format!("zstd: {e}"))),
            Codec::Bsc { coder, gpu } => {
                let p = bsc_sys::Params { coder, threads, gpu, ..Default::default() };
                bsc_sys::compress(data, &p).map_err(|e| Error::Codec(e.to_string()))
            }
            Codec::Ncore { gpu } => {
                // warmup() waits for the background start-up thread if it is still running.
                if gpu && ncore::gpu::warmup() {
                    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                    if let Ok((c, _)) = ncore::gpu::encode(data, NCORE_LANE) {
                        return Ok(c);
                    }
                }
                Ok(ncore::cpu::encode(data, NCORE_LANE))
            }
        }
    }
}

/// Decode `payload` produced by codec `id` into `out` (exact original length).
pub fn decompress(id: u8, payload: &[u8], out: &mut [u8], threads: usize, gpu: bool) -> Result<(), Error> {
    match id {
        CODEC_STORE => {
            if payload.len() != out.len() {
                return Err(Error::Corrupt("stored length mismatch"));
            }
            out.copy_from_slice(payload);
            Ok(())
        }
        CODEC_ZSTD => {
            let n = zstd::bulk::decompress_to_buffer(payload, out).map_err(|_| Error::Corrupt("zstd stream invalid"))?;
            if n != out.len() { Err(Error::Corrupt("zstd length mismatch")) } else { Ok(()) }
        }
        CODEC_BSC => bsc_sys::decompress(payload, out, threads, gpu).map_err(|_| Error::Corrupt("bsc stream invalid")),
        CODEC_NCORE => {
            // Decode straight into `out`. Any GPU failure falls back to the CPU reference.
            let gpu_ok = gpu && ncore::gpu::warmup() && {
                let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                ncore::gpu::decode_into(payload, out).is_ok()
            };
            if gpu_ok { Ok(()) } else { ncore::cpu::decode_into(payload, out).map_err(|_| Error::Corrupt("ncore stream invalid")) }
        }
        _ => Err(Error::Format("unknown codec")),
    }
}

/// Cheap test for data that won't compress (already-compressed media, archives,
/// encrypted data). Order-0 entropy on a few windows, confirmed with a zstd-1 trial
/// so a high-entropy-but-repetitive block is not misjudged.
pub fn looks_incompressible(data: &[u8]) -> bool {
    const WIN: usize = 4096;
    const SAMPLES: usize = 16;
    if data.len() < 64 * 1024 {
        return false;
    }
    let step = (data.len() - WIN) / (SAMPLES - 1);
    for s in 0..SAMPLES {
        let w = &data[s * step..s * step + WIN];
        let mut hist = [0u32; 256];
        for &b in w {
            hist[b as usize] += 1;
        }
        let h: f64 = hist.iter().filter(|&&c| c > 0).map(|&c| {
            let p = c as f64 / WIN as f64;
            -p * p.log2()
        }).sum();
        if h < 7.9 {
            return false;
        }
    }
    // Confirm on a larger contiguous sample.
    let n = data.len().min(1 << 20);
    let start = (data.len() - n) / 2;
    let sample = &data[start..start + n];
    match zstd::bulk::compress(sample, 1) {
        Ok(c) => c.len() as f64 > 0.98 * n as f64,
        Err(_) => false,
    }
}
