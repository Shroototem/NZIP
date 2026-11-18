//! CPU reference encoder/decoder (the specification). Lanes run in parallel with rayon.

use crate::model::O2Model;
use crate::stream::{self, Layout};
use crate::{quantize, rans, Error, FRAME};
use rayon::prelude::*;

/// Encode one lane: forward model pass per frame, then rANS backwards over the frame.
pub fn encode_lane(data: &[u8]) -> Vec<u16> {
    let mut m = O2Model::new();
    let mut r = [0u32; 256];
    let mut f = [0u32; 256];
    let mut out: Vec<u16> = Vec::with_capacity(data.len() / 2 + 8);
    let mut cf: Vec<(u32, u32)> = Vec::with_capacity(FRAME.min(data.len()));
    for frame in data.chunks(FRAME) {
        cf.clear();
        for &b in frame {
            m.weights(&mut r);
            quantize(&r, &mut f);
            let cum: u32 = f[..b as usize].iter().sum();
            cf.push((cum, f[b as usize]));
            m.update(b as u32);
        }
        let mut x = rans::L;
        let mut words: Vec<u16> = Vec::with_capacity(frame.len() / 2 + 2);
        for &(cum, freq) in cf.iter().rev() {
            rans::put(&mut x, cum, freq, &mut words);
        }
        rans::flush(x, &mut words);
        out.extend(words.iter().rev()); // decoder order
    }
    out
}

/// Decode one lane of `len` bytes into `out`.
pub fn decode_lane(words: &[u16], out: &mut [u8]) -> Result<(), Error> {
    let mut m = O2Model::new();
    let mut r = [0u32; 256];
    let mut f = [0u32; 256];
    let mut pos = 0usize;
    for frame in out.chunks_mut(FRAME) {
        let mut x = rans::init(words, &mut pos).ok_or(Error::Corrupt("truncated lane"))?;
        for o in frame.iter_mut() {
            m.weights(&mut r);
            quantize(&r, &mut f);
            let slot = rans::slot(x);
            let mut cum = 0u32;
            let mut s = 0usize;
            while cum + f[s] <= slot {
                cum += f[s];
                s += 1;
            }
            if !rans::advance(&mut x, cum, f[s], words, &mut pos) {
                return Err(Error::Corrupt("truncated lane"));
            }
            *o = s as u8;
            m.update(s as u32);
        }
        if x != rans::L {
            return Err(Error::Corrupt("bad final state"));
        }
    }
    if pos != words.len() {
        return Err(Error::Corrupt("unused words in lane"));
    }
    Ok(())
}

pub fn encode(data: &[u8], lane_len: usize) -> Vec<u8> {
    let layout = Layout::new(data.len(), lane_len);
    let lanes: Vec<Vec<u16>> = (0..layout.n_lanes).into_par_iter().map(|i| encode_lane(&data[layout.lane_range(i)])).collect();
    stream::write(&layout, &lanes)
}

pub fn decode(b: &[u8]) -> Result<Vec<u8>, Error> {
    let (layout, _, _) = stream::parse(b)?;
    let mut out = vec![0u8; layout.total];
    decode_into(b, &mut out)?;
    Ok(out)
}

/// Decode into `out`, which must be exactly the stream's original length.
pub fn decode_into(b: &[u8], out: &mut [u8]) -> Result<(), Error> {
    let (layout, offs, words) = stream::read(b)?;
    if out.len() != layout.total {
        return Err(Error::Corrupt("output length mismatch"));
    }
    out.par_chunks_mut(layout.lane_len)
        .enumerate()
        .try_for_each(|(i, o)| decode_lane(&words[offs[i]..offs[i + 1]], o))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn text(n: usize) -> Vec<u8> {
        let w = ["the ", "cat ", "sat ", "on ", "a ", "mat ", "\n", "and ", "then "];
        let mut x = 12345u32;
        let mut v = Vec::new();
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            v.extend_from_slice(w[(x % 9) as usize].as_bytes());
        }
        v.truncate(n);
        v
    }

    #[test]
    fn roundtrip_and_compresses() {
        let d = text(300_000);
        let c = encode(&d, 64 * 1024);
        assert!(c.len() < d.len() / 3, "{} vs {}", c.len(), d.len());
        assert_eq!(decode(&c).unwrap(), d);
    }

    #[test]
    fn multi_frame_lane() {
        let d = text(3 * crate::FRAME + 123);
        let c = encode(&d, d.len());
        assert_eq!(decode(&c).unwrap(), d);
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 48, .. ProptestConfig::default() })]
        #[test]
        fn roundtrip_any(d in proptest::collection::vec(any::<u8>(), 0..70_000), lane in 1usize..40_000) {
            let c = encode(&d, lane);
            prop_assert_eq!(decode(&c).unwrap(), d);
        }

        #[test]
        fn corruption_never_panics(seed in any::<u64>(), pos in 0.0f64..1.0, x in 1u8..=255) {
            let d = text(20_000 + (seed % 5000) as usize);
            let mut c = encode(&d, 4096);
            let p = ((c.len() - 1) as f64 * pos) as usize;
            c[p] ^= x;
            // Must return Err or (rarely, for bits that don't affect decoding) the
            // original; never panic. The container's checksums catch the rest.
            let _ = decode(&c);
        }
    }
}
