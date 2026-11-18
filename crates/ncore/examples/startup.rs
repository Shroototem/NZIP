//! Times the pieces of GPU start-up. Usage: startup <file>
use std::time::Instant;
fn main() {
    let data = std::fs::read(std::env::args().nth(1).expect("file")).unwrap();
    let t = Instant::now();
    let a = ncore::gpu::available();
    println!("available() [driver probe]: {:7.1} ms ({a})", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let w = ncore::gpu::warmup();
    println!("warmup() [context + module]: {:6.1} ms ({w})", t.elapsed().as_secs_f64() * 1e3);
    let enc = ncore::cpu::encode(&data, 64 * 1024);
    for i in 0..3 {
        let t = Instant::now();
        let (d, ms) = ncore::gpu::decode(&enc).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(d, data);
        println!("decode #{i}: wall {wall:6.1} ms, kernel {ms:6.1} ms, other {:5.1} ms", wall - ms as f64);
    }
    for i in 0..2 {
        let t = Instant::now();
        let (_c, ms) = ncore::gpu::encode(&data, 64 * 1024).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        println!("encode #{i}: wall {wall:6.1} ms, kernel {ms:6.1} ms, other {:5.1} ms", wall - ms as f64);
    }
}
