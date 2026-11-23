//! Model constants shared by the coders (the mixer itself lives in `cm`).

/// Model constants. They are part of the archive format: encoder and decoder must agree.
/// With `--features tune`, `NZ_K="i=v,..."` overrides them for experiments only.
pub fn knob(i: usize) -> i32 {
    const DEFAULTS: [i32; 4] = [
        2,   // 0: mixer learning rate
        7,   // 1: slow counter rate (shift)
        3,   // 2: fast counter rate (shift)
        0,
    ];
    #[cfg(feature = "tune")]
    {
        static K: std::sync::OnceLock<[i32; 4]> = std::sync::OnceLock::new();
        K.get_or_init(|| {
            let mut k = DEFAULTS;
            if let Ok(v) = std::env::var("NZ_K") {
                for kv in v.split(',') {
                    if let Some((a, b)) = kv.split_once('=') {
                        if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<i32>()) {
                            k[a] = b;
                        }
                    }
                }
            }
            k
        })[i]
    }
    #[cfg(not(feature = "tune"))]
    DEFAULTS[i]
}
