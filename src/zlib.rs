//! Raw deflate streams, both ways:
//!   * `inflate`: a plain decoder that also reports where the stream ends;
//!   * `deflate`: a bit-exact port of zlib 1.3.1's compressor (raw stream, 32 KB window,
//!     default strategy, levels 1-9, memLevel 1-9), so a stream zlib wrote can be rebuilt
//!     byte for byte from its inflated data and two small numbers.
//!
//! Every choice zlib makes (hash chains, lazy matching, window slides, block cuts, tree
//! building and tie breaks, stored / static / dynamic block choice) follows the C code
//! step for step; a different order anywhere gives a valid but different stream.
//! zlib (C) 1995-2024 Jean-loup Gailly and Mark Adler, zlib license.

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const WSIZE: usize = 1 << 15;
const WMASK: usize = WSIZE - 1;
const MIN_LOOKAHEAD: usize = MAX_MATCH + MIN_MATCH + 1;
const MAX_DIST: usize = WSIZE - MIN_LOOKAHEAD;
const WIN_INIT: usize = MAX_MATCH;
const TOO_FAR: u32 = 4096;

const LITERALS: usize = 256;
const END_BLOCK: usize = 256;
const LENGTH_CODES: usize = 29;
const L_CODES: usize = LITERALS + 1 + LENGTH_CODES;
const D_CODES: usize = 30;
const BL_CODES: usize = 19;
const HEAP_SIZE: usize = 2 * L_CODES + 1;
const MAX_BITS: usize = 15;
const MAX_BL_BITS: usize = 7;
const REP_3_6: usize = 16;
const REPZ_3_10: usize = 17;
const REPZ_11_138: usize = 18;

const EXTRA_LBITS: [u8; LENGTH_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const EXTRA_DBITS: [u8; D_CODES] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const EXTRA_BLBITS: [u8; BL_CODES] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
const BL_ORDER: [u8; BL_CODES] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// (good, lazy, nice, chain) per level; levels 1-3 are the greedy compressor.
const CONFIG: [(usize, usize, usize, usize); 10] = [
    (0, 0, 0, 0),
    (4, 4, 8, 4),
    (4, 5, 16, 8),
    (4, 6, 32, 32),
    (4, 4, 16, 16),
    (8, 16, 32, 32),
    (8, 16, 128, 128),
    (8, 32, 128, 256),
    (32, 128, 258, 1024),
    (32, 258, 258, 4096),
];

struct Tables {
    lcode: [u16; L_CODES + 2],
    llen: [u16; L_CODES + 2],
    dcode: [u16; D_CODES],
    dist_code: [u8; 512],
    length_code: [u8; 256],
    base_length: [u16; LENGTH_CODES],
    base_dist: [u16; D_CODES],
}

fn bi_reverse(mut code: u32, len: u32) -> u32 {
    let mut res = 0;
    for _ in 0..len {
        res = (res << 1) | (code & 1);
        code >>= 1;
    }
    res
}

/// Canonical codes from code lengths (zlib's gen_codes).
fn gen_codes(len: &[u16], code: &mut [u16], max_code: usize, bl_count: &[u16; MAX_BITS + 1]) {
    let mut next = [0u16; MAX_BITS + 1];
    let mut c = 0u32;
    for bits in 1..=MAX_BITS {
        c = (c + bl_count[bits - 1] as u32) << 1;
        next[bits] = c as u16;
    }
    for n in 0..=max_code {
        let l = len[n] as usize;
        if l != 0 {
            code[n] = bi_reverse(next[l] as u32, l as u32) as u16;
            next[l] = next[l].wrapping_add(1);
        }
    }
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            lcode: [0; L_CODES + 2],
            llen: [0; L_CODES + 2],
            dcode: [0; D_CODES],
            dist_code: [0; 512],
            length_code: [0; 256],
            base_length: [0; LENGTH_CODES],
            base_dist: [0; D_CODES],
        };
        let mut length = 0usize;
        let mut code = 0;
        while code < LENGTH_CODES - 1 {
            t.base_length[code] = length as u16;
            for _ in 0..1 << EXTRA_LBITS[code] {
                t.length_code[length] = code as u8;
                length += 1;
            }
            code += 1;
        }
        t.length_code[length - 1] = code as u8;
        let mut dist = 0usize;
        for code in 0..16 {
            t.base_dist[code] = dist as u16;
            for _ in 0..1 << EXTRA_DBITS[code] {
                t.dist_code[dist] = code as u8;
                dist += 1;
            }
        }
        dist >>= 7;
        for code in 16..D_CODES {
            t.base_dist[code] = (dist << 7) as u16;
            for _ in 0..1 << (EXTRA_DBITS[code] - 7) {
                t.dist_code[256 + dist] = code as u8;
                dist += 1;
            }
        }
        let mut bl_count = [0u16; MAX_BITS + 1];
        for n in 0..L_CODES + 2 {
            let l = match n {
                0..=143 => 8,
                144..=255 => 9,
                256..=279 => 7,
                _ => 8,
            };
            t.llen[n] = l;
            bl_count[l as usize] += 1;
        }
        let llen = t.llen;
        gen_codes(&llen, &mut t.lcode, L_CODES + 1, &bl_count);
        for n in 0..D_CODES {
            t.dcode[n] = bi_reverse(n as u32, 5) as u16;
        }
        t
    })
}

#[inline(always)]
fn d_code(t: &Tables, dist: usize) -> usize {
    (if dist < 256 { t.dist_code[dist] } else { t.dist_code[256 + (dist >> 7)] }) as usize
}

/// One Huffman tree under construction (zlib keeps freq/code and dad/len in unions; they
/// are never live at the same time, so separate arrays behave the same).
struct Tree {
    freq: [u16; HEAP_SIZE],
    code: [u16; HEAP_SIZE],
    len: [u16; HEAP_SIZE],
    dad: [u16; HEAP_SIZE],
    max_code: usize,
}

impl Tree {
    fn new() -> Box<Tree> {
        Box::new(Tree { freq: [0; HEAP_SIZE], code: [0; HEAP_SIZE], len: [0; HEAP_SIZE], dad: [0; HEAP_SIZE], max_code: 0 })
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Lit,
    Dist,
    Bl,
}

struct Builder {
    /// heap entries: (frequency << 8 | depth) << 16 | node; zlib's `smaller` is then one
    /// compare of the high bits (ties count as smaller, as in zlib)
    heap: [u64; HEAP_SIZE],
    heap_len: usize,
    heap_max: usize,
    depth: [u8; HEAP_SIZE],
    bl_count: [u16; MAX_BITS + 1],
    opt_len: u64,
    static_len: u64,
}

impl Builder {
    #[inline(always)]
    fn key(&self, tree: &Tree, n: usize) -> u64 {
        ((tree.freq[n] as u64) << 8 | self.depth[n] as u64) << 16 | n as u64
    }

    fn pqdownheap(&mut self, mut k: usize) {
        let v = self.heap[k];
        let kv = v >> 16;
        let mut j = k << 1;
        while j <= self.heap_len {
            if j < self.heap_len && self.heap[j + 1] >> 16 <= self.heap[j] >> 16 {
                j += 1;
            }
            if kv <= self.heap[j] >> 16 {
                break;
            }
            self.heap[k] = self.heap[j];
            k = j;
            j <<= 1;
        }
        self.heap[k] = v;
    }

    fn gen_bitlen(&mut self, tree: &mut Tree, kind: Kind) {
        let t = tables();
        let max_code = tree.max_code;
        let (extra, base, max_length): (&[u8], usize, usize) = match kind {
            Kind::Lit => (&EXTRA_LBITS, LITERALS + 1, MAX_BITS),
            Kind::Dist => (&EXTRA_DBITS, 0, MAX_BITS),
            Kind::Bl => (&EXTRA_BLBITS, 0, MAX_BL_BITS),
        };
        self.bl_count = [0; MAX_BITS + 1];
        let mut overflow = 0i32;
        tree.len[self.heap[self.heap_max] as usize & 0xFFFF] = 0;
        let mut h = self.heap_max + 1;
        while h < HEAP_SIZE {
            let n = self.heap[h] as usize & 0xFFFF;
            let mut bits = tree.len[tree.dad[n] as usize] as usize + 1;
            if bits > max_length {
                bits = max_length;
                overflow += 1;
            }
            tree.len[n] = bits as u16;
            h += 1;
            if n > max_code {
                continue;
            }
            self.bl_count[bits] += 1;
            let xbits = if n >= base { extra[n - base] as usize } else { 0 };
            let f = tree.freq[n] as u64;
            self.opt_len = self.opt_len.wrapping_add(f * (bits + xbits) as u64);
            match kind {
                Kind::Lit => self.static_len = self.static_len.wrapping_add(f * (t.llen[n] as usize + xbits) as u64),
                Kind::Dist => self.static_len = self.static_len.wrapping_add(f * (5 + xbits) as u64),
                Kind::Bl => {}
            }
        }
        if overflow == 0 {
            return;
        }
        loop {
            let mut bits = max_length - 1;
            while self.bl_count[bits] == 0 {
                bits -= 1;
            }
            self.bl_count[bits] -= 1;
            self.bl_count[bits + 1] += 2;
            self.bl_count[max_length] -= 1;
            overflow -= 2;
            if overflow <= 0 {
                break;
            }
        }
        let mut h = HEAP_SIZE;
        for bits in (1..=max_length).rev() {
            let mut n = self.bl_count[bits];
            while n != 0 {
                h -= 1;
                let m = self.heap[h] as usize & 0xFFFF;
                if m > max_code {
                    continue;
                }
                if tree.len[m] as usize != bits {
                    self.opt_len = self.opt_len.wrapping_add((bits as u64).wrapping_sub(tree.len[m] as u64).wrapping_mul(tree.freq[m] as u64));
                    tree.len[m] = bits as u16;
                }
                n -= 1;
            }
        }
    }

    fn build_tree(&mut self, tree: &mut Tree, kind: Kind) {
        let t = tables();
        let elems = match kind {
            Kind::Lit => L_CODES,
            Kind::Dist => D_CODES,
            Kind::Bl => BL_CODES,
        };
        let mut max_code: isize = -1;
        self.heap_len = 0;
        self.heap_max = HEAP_SIZE;
        for n in 0..elems {
            if tree.freq[n] != 0 {
                self.heap_len += 1;
                self.heap[self.heap_len] = n as u64;
                max_code = n as isize;
                self.depth[n] = 0;
            } else {
                tree.len[n] = 0;
            }
        }
        while self.heap_len < 2 {
            let node = if max_code < 2 {
                max_code += 1;
                max_code as usize
            } else {
                0
            };
            self.heap_len += 1;
            self.heap[self.heap_len] = node as u64;
            tree.freq[node] = 1;
            self.depth[node] = 0;
            self.opt_len = self.opt_len.wrapping_sub(1);
            match kind {
                Kind::Lit => self.static_len = self.static_len.wrapping_sub(t.llen[node] as u64),
                Kind::Dist => self.static_len = self.static_len.wrapping_sub(5),
                Kind::Bl => {}
            }
        }
        tree.max_code = max_code as usize;
        for i in 1..=self.heap_len {
            let n = self.heap[i] as usize;
            self.heap[i] = self.key(tree, n);
        }
        for n in (1..=self.heap_len / 2).rev() {
            self.pqdownheap(n);
        }
        let mut node = elems;
        loop {
            let n = self.heap[1] as usize & 0xFFFF;
            self.heap[1] = self.heap[self.heap_len];
            self.heap_len -= 1;
            self.pqdownheap(1);
            let m = self.heap[1] as usize & 0xFFFF;
            self.heap_max -= 1;
            self.heap[self.heap_max] = n as u64;
            self.heap_max -= 1;
            self.heap[self.heap_max] = m as u64;
            tree.freq[node] = tree.freq[n].wrapping_add(tree.freq[m]);
            self.depth[node] = self.depth[n].max(self.depth[m]) + 1;
            tree.dad[n] = node as u16;
            tree.dad[m] = node as u16;
            self.heap[1] = self.key(tree, node);
            node += 1;
            self.pqdownheap(1);
            if self.heap_len < 2 {
                break;
            }
        }
        self.heap_max -= 1;
        self.heap[self.heap_max] = self.heap[1];
        self.gen_bitlen(tree, kind);
        let (len, code) = (&tree.len, &mut tree.code);
        gen_codes(len, code, tree.max_code, &self.bl_count);
    }
}

/// zlib's 16-bit bit buffer.
/// LSB-first bit writer. zlib keeps 16 bits and flushes pairs of bytes; a wider buffer
/// writes the same bit stream, so the same bytes.
struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    #[inline(always)]
    fn put_short(&mut self, w: u32) {
        self.out.extend_from_slice(&[w as u8, (w >> 8) as u8]);
    }
    #[inline(always)]
    fn send(&mut self, value: u32, length: u32) {
        self.acc |= (value as u64) << self.n;
        self.n += length;
        if self.n >= 32 {
            self.out.extend_from_slice(&(self.acc as u32).to_le_bytes());
            self.acc >>= 32;
            self.n -= 32;
        }
    }
    /// Pads to a byte boundary (zlib's bi_windup).
    fn windup(&mut self) {
        while self.n > 0 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n = self.n.saturating_sub(8);
        }
        self.acc = 0;
    }
}

struct State<'a> {
    input: &'a [u8],
    in_pos: usize,
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    hash_shift: u32,
    hash_mask: usize,
    ins_h: usize,
    strstart: usize,
    block_start: i64,
    lookahead: usize,
    insert: usize,
    match_length: usize,
    prev_match: u32,
    match_available: bool,
    match_start: u32,
    prev_length: usize,
    max_chain: usize,
    max_lazy: usize,
    good_match: usize,
    nice_match: usize,
    high_water: usize,
    sym: Vec<(u16, u8)>,
    sym_end: usize,
    ltree: Box<Tree>,
    dtree: Box<Tree>,
    bltree: Box<Tree>,
    b: Box<Builder>,
    bits: Bits,
    /// when set, stop as soon as the output stops matching this stream
    target: Option<&'a [u8]>,
    checked: usize,
    bad: bool,
    /// stream offset of `input[0]` (a segment starts with up to WSIZE bytes of history)
    base: usize,
    /// when recording: every clean block boundary as (stream offset, output bit offset)
    points: Option<Vec<(usize, u64)>>,
    /// stop at the clean block boundary at this stream offset (usize::MAX: never)
    stop: usize,
    stopped: bool,
}

const WINDOW_SIZE: usize = 2 * WSIZE;

/// Buffers one compressor needs, kept per thread: many zip entries are a few KB, and
/// allocating ~300 KB per stream would cost more than compressing it.
struct Scratch {
    window: Vec<u8>,
    prev: Vec<u16>,
    head: Vec<u16>,
    sym: Vec<(u16, u8)>,
    ltree: Box<Tree>,
    dtree: Box<Tree>,
    bltree: Box<Tree>,
    b: Box<Builder>,
}

thread_local! {
    static SCRATCH: std::cell::Cell<Option<Box<Scratch>>> = const { std::cell::Cell::new(None) };
}

fn take_scratch() -> Box<Scratch> {
    SCRATCH.with(|c| c.take()).unwrap_or_else(|| {
        Box::new(Scratch {
            window: vec![0; WINDOW_SIZE + 8],
            prev: vec![0; WSIZE],
            head: vec![0; 1 << 16],
            sym: Vec::with_capacity(1 << 15),
            ltree: Tree::new(),
            dtree: Tree::new(),
            bltree: Tree::new(),
            b: Box::new(Builder { heap: [0; HEAP_SIZE], heap_len: 0, heap_max: 0, depth: [0; HEAP_SIZE], bl_count: [0; MAX_BITS + 1], opt_len: 0, static_len: 0 }),
        })
    })
}

fn put_scratch(s: Box<Scratch>) {
    SCRATCH.with(|c| c.set(Some(s)));
}

impl<'a> State<'a> {
    fn new(input: &'a [u8], level: usize, mem: usize) -> State<'a> {
        let hash_bits = mem + 7;
        let lit_bufsize = 1usize << (mem + 6);
        let (good, lazy, nice, chain) = CONFIG[level];
        let sc = take_scratch();
        let Scratch { window, prev, mut head, mut sym, ltree, dtree, bltree, b } = *sc;
        // zlib clears the hash heads; the window and prev[] start as garbage that never
        // reaches the output, so reused buffers give the same streams
        head[..1 << hash_bits].fill(0);
        sym.clear();
        let mut s = State {
            input,
            in_pos: 0,
            window,
            prev,
            head,
            hash_shift: (hash_bits as u32 + MIN_MATCH as u32 - 1) / MIN_MATCH as u32,
            hash_mask: (1 << hash_bits) - 1,
            ins_h: 0,
            strstart: 0,
            block_start: 0,
            lookahead: 0,
            insert: 0,
            match_length: MIN_MATCH - 1,
            prev_match: 0,
            match_available: false,
            match_start: 0,
            prev_length: MIN_MATCH - 1,
            max_chain: chain,
            max_lazy: lazy,
            good_match: good,
            nice_match: nice,
            high_water: 0,
            sym,
            sym_end: lit_bufsize - 1,
            ltree,
            dtree,
            bltree,
            b,
            bits: Bits { out: Vec::with_capacity(input.len() / 2 + 64), acc: 0, n: 0 },
            target: None,
            checked: 0,
            bad: false,
            base: 0,
            points: None,
            stop: usize::MAX,
            stopped: false,
        };
        s.init_block();
        s
    }

    /// A compressor that starts at stream offset `start` of `data` at a clean block
    /// boundary (see `deflate_points`): the WSIZE bytes before it are loaded the way
    /// deflateSetDictionary loads a dictionary, which leaves the window, hash chains and
    /// block state as a run from the stream start has them there.
    fn new_at(data: &'a [u8], start: usize, level: usize, mem: usize) -> State<'a> {
        let h = start.min(WSIZE);
        let mut s = State::new(&data[start - h..], level, mem);
        s.base = start - h;
        s.window[..h].copy_from_slice(&data[start - h..start]);
        s.strstart = h;
        s.block_start = h as i64;
        s.in_pos = h;
        s.insert = h;
        s
    }

    /// Stream offset of `strstart`.
    #[inline(always)]
    fn at(&self) -> usize {
        self.base + self.in_pos - self.lookahead
    }

    /// Returns the output and hands the buffers back to this thread's scratch.
    fn finish(self) -> (Vec<u8>, bool) {
        let (out, _, bad) = self.finish_bits();
        (out, bad)
    }

    /// The output with any bits not yet written (zero-padded to a byte), its length in
    /// bits, and the mismatch flag.
    fn finish_bits(self) -> (Vec<u8>, u64, bool) {
        let State { window, prev, head, sym, ltree, dtree, bltree, b, mut bits, bad, .. } = self;
        put_scratch(Box::new(Scratch { window, prev, head, sym, ltree, dtree, bltree, b }));
        let nbits = bits.out.len() as u64 * 8 + bits.n as u64;
        let tail = bits.acc.to_le_bytes();
        bits.out.extend_from_slice(&tail[..bits.n.div_ceil(8) as usize]);
        (bits.out, nbits, bad)
    }

    fn init_block(&mut self) {
        self.ltree.freq[..L_CODES].fill(0);
        self.dtree.freq[..D_CODES].fill(0);
        self.bltree.freq[..BL_CODES].fill(0);
        self.ltree.freq[END_BLOCK] = 1;
        self.b.opt_len = 0;
        self.b.static_len = 0;
        self.sym.clear();
    }

    #[inline(always)]
    fn update_hash(&mut self, c: u8) {
        self.ins_h = ((self.ins_h << self.hash_shift) ^ c as usize) & self.hash_mask;
    }

    /// INSERT_STRING: returns the previous head of the chain.
    #[inline(always)]
    fn insert_string(&mut self, s: usize) -> usize {
        let c = self.window[s + MIN_MATCH - 1];
        self.update_hash(c);
        // ins_h is masked to the hash size (<= head.len()); s & WMASK < prev.len()
        unsafe {
            let h = *self.head.get_unchecked(self.ins_h);
            *self.prev.get_unchecked_mut(s & WMASK) = h;
            *self.head.get_unchecked_mut(self.ins_h) = s as u16;
            h as usize
        }
    }

    fn slide_hash(&mut self) {
        for m in self.head.iter_mut().chain(self.prev.iter_mut()) {
            *m = if *m as usize >= WSIZE { *m - WSIZE as u16 } else { 0 };
        }
    }

    fn fill_window(&mut self) {
        loop {
            let mut more = WINDOW_SIZE - self.lookahead - self.strstart;
            if self.strstart >= WSIZE + MAX_DIST {
                self.window.copy_within(WSIZE..WSIZE + WSIZE - more, 0);
                self.match_start = self.match_start.wrapping_sub(WSIZE as u32);
                self.strstart -= WSIZE;
                self.block_start -= WSIZE as i64;
                if self.insert > self.strstart {
                    self.insert = self.strstart;
                }
                self.slide_hash();
                more += WSIZE;
            }
            let avail = self.input.len() - self.in_pos;
            if avail == 0 {
                break;
            }
            let n = avail.min(more);
            let at = self.strstart + self.lookahead;
            self.window[at..at + n].copy_from_slice(&self.input[self.in_pos..self.in_pos + n]);
            self.in_pos += n;
            self.lookahead += n;
            if self.lookahead + self.insert >= MIN_MATCH {
                let mut st = self.strstart - self.insert;
                self.ins_h = self.window[st] as usize;
                let c = self.window[st + 1];
                self.update_hash(c);
                while self.insert != 0 {
                    let c = self.window[st + MIN_MATCH - 1];
                    self.update_hash(c);
                    self.prev[st & WMASK] = self.head[self.ins_h];
                    self.head[self.ins_h] = st as u16;
                    st += 1;
                    self.insert -= 1;
                    if self.lookahead + self.insert < MIN_MATCH {
                        break;
                    }
                }
            }
            if !(self.lookahead < MIN_LOOKAHEAD && self.in_pos < self.input.len()) {
                break;
            }
        }
        if self.high_water < WINDOW_SIZE {
            let curr = self.strstart + self.lookahead;
            if self.high_water < curr {
                let init = (WINDOW_SIZE - curr).min(WIN_INIT);
                self.window[curr..curr + init].fill(0);
                self.high_water = curr + init;
            } else if self.high_water < curr + WIN_INIT {
                let init = (curr + WIN_INIT - self.high_water).min(WINDOW_SIZE - self.high_water);
                self.window[self.high_water..self.high_water + init].fill(0);
                self.high_water += init;
            }
        }
    }

    /// zlib's longest_match (byte-compare version): the longest match on the chain from
    /// `cur`, longer than prev_length; sets match_start.
    #[inline(always)]
    fn longest_match(&mut self, mut cur: usize) -> usize {
        let w = &self.window[..];
        let mut chain = self.max_chain;
        let scan = self.strstart;
        let mut best_len = self.prev_length;
        let mut nice = self.nice_match;
        let limit = if self.strstart > MAX_DIST { self.strstart - MAX_DIST } else { 0 };
        let mut end16 = u16::from_le_bytes([w[scan + best_len - 1], w[scan + best_len]]);
        if self.prev_length >= self.good_match {
            chain >>= 2;
        }
        if nice > self.lookahead {
            nice = self.lookahead;
        }
        let s16 = u16::from_le_bytes([w[scan], w[scan + 1]]);
        loop {
            let m = cur;
            // m < scan and best_len <= MAX_MATCH: in bounds of the padded window
            let r16 = |i: usize| unsafe { (w.as_ptr().add(i) as *const u16).read_unaligned() }.to_le();
            if r16(m + best_len - 1) == end16 && r16(m) == s16 {
                // bytes 0, 1 match; byte 2 is implied by the hash (never compared); the
                // length is the first mismatch at index 3..=258, capped at 258, as zlib's
                // byte loop gives, found 8 bytes at a time (the window has 8 spare bytes)
                let rd = |i: usize| unsafe { (w.as_ptr().add(i) as *const u64).read_unaligned() }.to_le();
                let mut i = 3;
                let len = loop {
                    let x = rd(scan + i) ^ rd(m + i);
                    if x != 0 {
                        break (i + (x.trailing_zeros() / 8) as usize).min(MAX_MATCH);
                    }
                    i += 8;
                    if i > MAX_MATCH {
                        break MAX_MATCH;
                    }
                };
                if len > best_len {
                    self.match_start = cur as u32;
                    best_len = len;
                    if len >= nice {
                        break;
                    }
                    end16 = u16::from_le_bytes([w[scan + best_len - 1], w[scan + best_len]]);
                }
            }
            cur = unsafe { *self.prev.get_unchecked(cur & WMASK) } as usize;
            if cur <= limit {
                break;
            }
            chain -= 1;
            if chain == 0 {
                break;
            }
        }
        best_len.min(self.lookahead)
    }

    #[inline(always)]
    fn tally_lit(&mut self, c: u8) -> bool {
        self.sym.push((0, c));
        self.ltree.freq[c as usize] += 1;
        self.sym.len() == self.sym_end
    }

    #[inline(always)]
    fn tally_dist(&mut self, dist: usize, lc: usize) -> bool {
        let t = tables();
        self.sym.push((dist as u16, lc as u8));
        self.ltree.freq[t.length_code[lc] as usize + LITERALS + 1] += 1;
        self.dtree.freq[d_code(t, dist - 1)] += 1;
        self.sym.len() == self.sym_end
    }

    fn flush_block(&mut self, last: bool) {
        let buf = (self.block_start >= 0).then_some(self.block_start as usize);
        let stored_len = (self.strstart as i64 - self.block_start) as u64;
        self.tr_flush_block(buf, stored_len, last);
        self.block_start = self.strstart as i64;
        if let Some(t) = self.target {
            let o = &self.bits.out;
            if o.len() > t.len() || o[self.checked..] != t[self.checked..o.len()] {
                self.bad = true;
            }
            self.checked = o.len();
        }
    }

    fn scan_tree(&mut self, which: Kind) {
        let (tree, bl) = match which {
            Kind::Lit => (&mut self.ltree, &mut self.bltree),
            _ => (&mut self.dtree, &mut self.bltree),
        };
        let max_code = tree.max_code;
        let mut prevlen: i32 = -1;
        let mut nextlen = tree.len[0] as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        tree.len[max_code + 1] = 0xFFFF;
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = tree.len[n + 1] as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                bl.freq[curlen as usize] += count as u16;
            } else if curlen != 0 {
                if curlen != prevlen {
                    bl.freq[curlen as usize] += 1;
                }
                bl.freq[REP_3_6] += 1;
            } else if count <= 10 {
                bl.freq[REPZ_3_10] += 1;
            } else {
                bl.freq[REPZ_11_138] += 1;
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    fn send_tree(bits: &mut Bits, tree: &Tree, bl: &Tree, max_code: usize) {
        let mut prevlen: i32 = -1;
        let mut nextlen = tree.len[0] as i32;
        let mut count = 0;
        let (mut max_count, mut min_count) = if nextlen == 0 { (138, 3) } else { (7, 4) };
        let send_code = |bits: &mut Bits, c: usize| bits.send(bl.code[c] as u32, bl.len[c] as u32);
        for n in 0..=max_code {
            let curlen = nextlen;
            nextlen = tree.len[n + 1] as i32;
            count += 1;
            if count < max_count && curlen == nextlen {
                continue;
            } else if count < min_count {
                while count != 0 {
                    send_code(bits, curlen as usize);
                    count -= 1;
                }
            } else if curlen != 0 {
                if curlen != prevlen {
                    send_code(bits, curlen as usize);
                    count -= 1;
                }
                send_code(bits, REP_3_6);
                bits.send(count as u32 - 3, 2);
            } else if count <= 10 {
                send_code(bits, REPZ_3_10);
                bits.send(count as u32 - 3, 3);
            } else {
                send_code(bits, REPZ_11_138);
                bits.send(count as u32 - 11, 7);
            }
            count = 0;
            prevlen = curlen;
            (max_count, min_count) = if nextlen == 0 {
                (138, 3)
            } else if curlen == nextlen {
                (6, 3)
            } else {
                (7, 4)
            };
        }
    }

    fn compress_block(&mut self, dynamic: bool) {
        let t = tables();
        let (lc_, ll_, dc_, dl_): (&[u16], &[u16], &[u16], &[u16]) = if dynamic {
            (&self.ltree.code, &self.ltree.len, &self.dtree.code, &self.dtree.len)
        } else {
            static FIVE: [u16; D_CODES] = [5; D_CODES];
            (&t.lcode, &t.llen, &t.dcode, &FIVE)
        };
        let bits = &mut self.bits;
        for &(dist, lc) in &self.sym {
            let lc = lc as usize;
            if dist == 0 {
                bits.send(lc_[lc] as u32, ll_[lc] as u32);
            } else {
                let code = t.length_code[lc] as usize;
                bits.send(lc_[code + LITERALS + 1] as u32, ll_[code + LITERALS + 1] as u32);
                let extra = EXTRA_LBITS[code] as u32;
                if extra != 0 {
                    bits.send((lc - t.base_length[code] as usize) as u32, extra);
                }
                let d = dist as usize - 1;
                let code = d_code(t, d);
                bits.send(dc_[code] as u32, dl_[code] as u32);
                let extra = EXTRA_DBITS[code] as u32;
                if extra != 0 {
                    bits.send((d - t.base_dist[code] as usize) as u32, extra);
                }
            }
        }
        bits.send(lc_[END_BLOCK] as u32, ll_[END_BLOCK] as u32);
    }

    fn tr_flush_block(&mut self, buf: Option<usize>, stored_len: u64, last: bool) {
        self.b.build_tree(&mut self.ltree, Kind::Lit);
        self.b.build_tree(&mut self.dtree, Kind::Dist);
        // build_bl_tree
        self.scan_tree(Kind::Lit);
        self.scan_tree(Kind::Dist);
        self.b.build_tree(&mut self.bltree, Kind::Bl);
        let mut max_blindex = BL_CODES - 1;
        while max_blindex >= 3 {
            if self.bltree.len[BL_ORDER[max_blindex] as usize] != 0 {
                break;
            }
            max_blindex -= 1;
        }
        self.b.opt_len = self.b.opt_len.wrapping_add(3 * (max_blindex as u64 + 1) + 5 + 5 + 4);
        let mut opt_lenb = self.b.opt_len.wrapping_add(3 + 7) >> 3;
        let static_lenb = self.b.static_len.wrapping_add(3 + 7) >> 3;
        if static_lenb <= opt_lenb {
            opt_lenb = static_lenb;
        }
        let last_bit = last as u32;
        if stored_len + 4 <= opt_lenb && buf.is_some() {
            let at = buf.unwrap();
            self.bits.send(last_bit, 3);
            self.bits.windup();
            self.bits.put_short(stored_len as u32 & 0xFFFF);
            self.bits.put_short(!(stored_len as u32) & 0xFFFF);
            self.bits.out.extend_from_slice(&self.window[at..at + stored_len as usize]);
        } else if static_lenb == opt_lenb {
            self.bits.send((1 << 1) + last_bit, 3);
            self.compress_block(false);
        } else {
            self.bits.send((2 << 1) + last_bit, 3);
            let (lcodes, dcodes, blcodes) = (self.ltree.max_code + 1, self.dtree.max_code + 1, max_blindex + 1);
            self.bits.send(lcodes as u32 - 257, 5);
            self.bits.send(dcodes as u32 - 1, 5);
            self.bits.send(blcodes as u32 - 4, 4);
            for rank in 0..blcodes {
                self.bits.send(self.bltree.len[BL_ORDER[rank] as usize] as u32, 3);
            }
            Self::send_tree(&mut self.bits, &self.ltree, &self.bltree, lcodes - 1);
            Self::send_tree(&mut self.bits, &self.dtree, &self.bltree, dcodes - 1);
            self.compress_block(true);
        }
        self.init_block();
        if last {
            self.bits.windup();
        }
    }

    fn deflate_fast(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            if hash_head != 0 && self.strstart - hash_head <= MAX_DIST {
                self.match_length = self.longest_match(hash_head);
            }
            let bflush;
            if self.match_length >= MIN_MATCH {
                bflush = self.tally_dist(self.strstart - self.match_start as usize, self.match_length - MIN_MATCH);
                self.lookahead -= self.match_length;
                if self.match_length <= self.max_lazy && self.lookahead >= MIN_MATCH {
                    self.match_length -= 1;
                    loop {
                        self.strstart += 1;
                        self.insert_string(self.strstart);
                        self.match_length -= 1;
                        if self.match_length == 0 {
                            break;
                        }
                    }
                    self.strstart += 1;
                } else {
                    self.strstart += self.match_length;
                    self.match_length = 0;
                    self.ins_h = self.window[self.strstart] as usize;
                    let c = self.window[self.strstart + 1];
                    self.update_hash(c);
                }
            } else {
                bflush = self.tally_lit(self.window[self.strstart]);
                self.lookahead -= 1;
                self.strstart += 1;
            }
            if bflush {
                self.flush_block(false);
                if self.bad {
                    return;
                }
            }
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(true);
    }

    fn deflate_slow(&mut self) {
        loop {
            if self.lookahead < MIN_LOOKAHEAD {
                self.fill_window();
                if self.lookahead == 0 {
                    break;
                }
            }
            let mut hash_head = 0;
            if self.lookahead >= MIN_MATCH {
                hash_head = self.insert_string(self.strstart);
            }
            self.prev_length = self.match_length;
            self.prev_match = self.match_start;
            self.match_length = MIN_MATCH - 1;
            if hash_head != 0 && self.prev_length < self.max_lazy && self.strstart - hash_head <= MAX_DIST {
                self.match_length = self.longest_match(hash_head);
                if self.match_length == MIN_MATCH && (self.strstart as u32).wrapping_sub(self.match_start) > TOO_FAR {
                    self.match_length = MIN_MATCH - 1;
                }
            }
            if self.prev_length >= MIN_MATCH && self.match_length <= self.prev_length {
                let max_insert = self.strstart + self.lookahead - MIN_MATCH;
                let dist = (self.strstart as u32 - 1).wrapping_sub(self.prev_match) as usize;
                let bflush = self.tally_dist(dist, self.prev_length - MIN_MATCH);
                self.lookahead -= self.prev_length - 1;
                self.prev_length -= 2;
                loop {
                    self.strstart += 1;
                    if self.strstart <= max_insert {
                        self.insert_string(self.strstart);
                    }
                    self.prev_length -= 1;
                    if self.prev_length == 0 {
                        break;
                    }
                }
                self.match_available = false;
                self.match_length = MIN_MATCH - 1;
                self.strstart += 1;
                if bflush {
                    self.flush_block(false);
                    if self.bad {
                        return;
                    }
                    // a clean boundary: no pending lazy match, so the next block depends
                    // only on the WSIZE bytes before it
                    let at = self.at();
                    if let Some(p) = self.points.as_mut() {
                        p.push((at, self.bits.out.len() as u64 * 8 + self.bits.n as u64));
                    }
                    if at == self.stop {
                        self.stopped = true;
                        return;
                    }
                }
            } else if self.match_available {
                let bflush = self.tally_lit(self.window[self.strstart - 1]);
                if bflush {
                    self.flush_block(false);
                    if self.bad {
                        return;
                    }
                }
                self.strstart += 1;
                self.lookahead -= 1;
            } else {
                self.match_available = true;
                self.strstart += 1;
                self.lookahead -= 1;
            }
        }
        if self.match_available {
            self.tally_lit(self.window[self.strstart - 1]);
            self.match_available = false;
        }
        self.insert = self.strstart.min(MIN_MATCH - 1);
        self.flush_block(true);
    }
}

/// Raw deflate of `data` exactly as zlib 1.3.1 writes it with deflateInit2(level, Z_DEFLATED,
/// -15, mem, Z_DEFAULT_STRATEGY) and one deflate(Z_FINISH) call. `level` 1..=9, `mem` 1..=9.
pub fn deflate(data: &[u8], level: u8, mem: u8) -> Vec<u8> {
    let (level, mem) = (level.clamp(1, 9) as usize, mem.clamp(1, 9) as usize);
    let mut s = State::new(data, level, mem);
    if level <= 3 {
        s.deflate_fast();
    } else {
        s.deflate_slow();
    }
    s.finish().0
}

/// True when `deflate(data, level, mem)` would equal `target`; stops at the first block
/// that differs, so a wrong guess costs about one block of work.
pub fn deflate_eq(data: &[u8], level: u8, mem: u8, target: &[u8]) -> bool {
    let (level, mem) = (level.clamp(1, 9) as usize, mem.clamp(1, 9) as usize);
    let mut s = State::new(data, level, mem);
    s.target = Some(target);
    if level <= 3 {
        s.deflate_fast();
    } else {
        s.deflate_slow();
    }
    let (out, bad) = s.finish();
    !bad && out == target
}

/// Like `deflate_eq` (levels 4..=9), also returning the stream's clean block boundaries
/// (stream offset, output bit offset): points where `deflate_seg` can start or stop.
pub fn deflate_points(data: &[u8], level: u8, mem: u8, target: &[u8]) -> Option<Vec<(usize, u64)>> {
    let (level, mem) = (level.clamp(4, 9) as usize, mem.clamp(1, 9) as usize);
    let mut s = State::new(data, level, mem);
    s.target = Some(target);
    s.points = Some(Vec::new());
    s.deflate_slow();
    let pts = s.points.take().unwrap_or_default();
    let (out, bad) = s.finish();
    (!bad && out == target).then_some(pts)
}

/// The part of `deflate(data, level, mem)` (levels 4..=9) from the clean boundary at
/// stream offset `start` (0 = the stream start) to the one at `stop` (None = the end),
/// written after `lead` (< 8) zero bits so its bytes line up with the whole stream's (the
/// boundary's bit offset mod 8): its bytes, zero-padded, and its length in bits counting
/// the lead. None if `stop` is not a boundary.
pub fn deflate_seg(data: &[u8], level: u8, mem: u8, start: usize, stop: Option<usize>, lead: u32) -> Option<(Vec<u8>, u64)> {
    let (level, mem) = (level.clamp(4, 9) as usize, mem.clamp(1, 9) as usize);
    if start > data.len() || lead >= 8 {
        return None;
    }
    let mut s = State::new_at(data, start, level, mem);
    s.bits.n = lead;
    s.stop = stop.unwrap_or(usize::MAX);
    s.deflate_slow();
    if stop.is_some() != s.stopped {
        return None;
    }
    let (out, nbits, _) = s.finish_bits();
    Some((out, nbits))
}

/// Appends a segment from `deflate_seg` (written with lead = `*len` mod 8, `bits` bits
/// counting the lead) to the bit stream of `*len` bits in `out`.
pub fn append_bits(out: &mut Vec<u8>, len: &mut u64, seg: &[u8], bits: u64) {
    let lead = *len & 7;
    let seg = &seg[..bits.div_ceil(8) as usize];
    out.truncate(len.div_ceil(8) as usize);
    if lead == 0 {
        out.extend_from_slice(seg);
    } else if let Some((&first, rest)) = seg.split_first() {
        *out.last_mut().unwrap() |= first;
        out.extend_from_slice(rest);
    }
    *len += bits - lead;
    out.truncate(len.div_ceil(8) as usize);
}

// ---------------------------------------------------------------- inflate

struct Huff {
    /// fast table: (symbol << 4) | length, for codes up to FAST bits; 0 = longer code
    fast: Vec<u16>,
    /// canonical decoding for long codes
    count: [u16; 16],
    symbols: Vec<u16>,
}

const FAST: u32 = 10;

impl Huff {
    fn new(lens: &[u8]) -> Option<Huff> {
        let mut count = [0u16; 16];
        for &l in lens {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut left = 1i32;
        for c in &count[1..] {
            left = (left << 1) - *c as i32;
            if left < 0 {
                return None;
            }
        }
        let mut offs = [0u16; 16];
        for i in 1..15 {
            offs[i + 1] = offs[i] + count[i];
        }
        let mut symbols = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        let mut fast = vec![0u16; 1 << FAST];
        let mut code = 0u32;
        let mut k = 0usize;
        for len in 1..=15u32 {
            for _ in 0..count[len as usize] {
                if len <= FAST {
                    let r = bi_reverse(code, len);
                    let mut i = r as usize;
                    while i < 1 << FAST {
                        fast[i] = ((symbols[k] as u16) << 4) | len as u16;
                        i += 1 << len;
                    }
                }
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        Some(Huff { fast, count, symbols })
    }
}

struct In<'a> {
    src: &'a [u8],
    pos: usize,
    bit: u64,
    n: u32,
}

impl In<'_> {
    #[inline(always)]
    fn need(&mut self, k: u32) -> Option<()> {
        while self.n < k {
            let b = *self.src.get(self.pos)?;
            self.pos += 1;
            self.bit |= (b as u64) << self.n;
            self.n += 8;
        }
        Some(())
    }
    #[inline(always)]
    fn refill(&mut self) {
        if self.pos + 8 <= self.src.len() {
            while self.n <= 56 {
                self.bit |= (self.src[self.pos] as u64) << self.n;
                self.pos += 1;
                self.n += 8;
            }
        }
    }
    #[inline(always)]
    fn bits(&mut self, k: u32) -> Option<u32> {
        if k == 0 {
            return Some(0);
        }
        self.need(k)?;
        let v = (self.bit & ((1 << k) - 1)) as u32;
        self.bit >>= k;
        self.n -= k;
        Some(v)
    }
    #[inline(always)]
    fn sym(&mut self, h: &Huff) -> Option<u16> {
        self.refill();
        while self.n < 15 && self.pos < self.src.len() {
            // near the end: take what is there
            self.bit |= (self.src[self.pos] as u64) << self.n;
            self.pos += 1;
            self.n += 8;
        }
        let e = h.fast[(self.bit & ((1 << FAST) - 1)) as usize];
        if e != 0 {
            let l = (e & 15) as u32;
            if l > self.n {
                return None;
            }
            self.bit >>= l;
            self.n -= l;
            return Some(e >> 4);
        }
        // slow canonical walk
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=15 {
            code |= self.bits(1)? as i32;
            let c = h.count[len] as i32;
            if code - c < first {
                return Some(h.symbols[(index + (code - first)) as usize]);
            }
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        None
    }
}

const LBASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const DBASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];

/// Decodes the raw deflate stream at the start of `src`. Returns the data and the number
/// of bytes the stream occupies, or None if it is not a valid stream or the output would
/// exceed `limit`.
pub fn inflate(src: &[u8], limit: usize) -> Option<(Vec<u8>, usize)> {
    let mut r = In { src, pos: 0, bit: 0, n: 0 };
    let mut out: Vec<u8> = Vec::with_capacity((src.len() * 3).min(limit).min(1 << 20));
    loop {
        let last = r.bits(1)?;
        match r.bits(2)? {
            0 => {
                // stored: drop to a byte boundary
                let drop = r.n % 8;
                r.bits(drop)?;
                let len = r.bits(16)?;
                let nlen = r.bits(16)?;
                if len != !nlen & 0xFFFF {
                    return None;
                }
                // whole bytes still in the bit buffer come first
                let mut len = len as usize;
                while len > 0 && r.n >= 8 {
                    out.push(r.bits(8)? as u8);
                    len -= 1;
                }
                let s = src.get(r.pos..r.pos + len)?;
                out.extend_from_slice(s);
                r.pos += len;
            }
            t @ (1 | 2) => {
                let (lit, dist) = if t == 1 {
                    let mut l = [0u8; 288];
                    l[..144].fill(8);
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    l[280..].fill(8);
                    (Huff::new(&l)?, Huff::new(&[5u8; 30])?)
                } else {
                    let hlit = r.bits(5)? as usize + 257;
                    let hdist = r.bits(5)? as usize + 1;
                    let hclen = r.bits(4)? as usize + 4;
                    let mut cl = [0u8; 19];
                    for &o in &BL_ORDER[..hclen] {
                        cl[o as usize] = r.bits(3)? as u8;
                    }
                    let ch = Huff::new(&cl)?;
                    let mut lens = vec![0u8; hlit + hdist];
                    let mut i = 0;
                    while i < hlit + hdist {
                        let s = r.sym(&ch)?;
                        match s {
                            0..=15 => {
                                lens[i] = s as u8;
                                i += 1;
                            }
                            16 => {
                                let p = *lens.get(i.checked_sub(1)?)?;
                                for _ in 0..3 + r.bits(2)? {
                                    *lens.get_mut(i)? = p;
                                    i += 1;
                                }
                            }
                            17 => i += 3 + r.bits(3)? as usize,
                            _ => i += 11 + r.bits(7)? as usize,
                        }
                    }
                    if i > hlit + hdist || lens[256] == 0 {
                        return None;
                    }
                    (Huff::new(&lens[..hlit])?, Huff::new(&lens[hlit..])?)
                };
                loop {
                    let s = r.sym(&lit)? as usize;
                    if s < 256 {
                        out.push(s as u8);
                        continue;
                    }
                    if s == 256 {
                        break;
                    }
                    let k = s - 257;
                    if k >= 29 {
                        return None;
                    }
                    let len = LBASE[k] as usize + r.bits(EXTRA_LBITS[k] as u32)? as usize;
                    let ds = r.sym(&dist)? as usize;
                    if ds >= 30 {
                        return None;
                    }
                    let d = DBASE[ds] as usize + r.bits(EXTRA_DBITS[ds] as u32)? as usize;
                    if d > out.len() {
                        return None;
                    }
                    let st = out.len() - d;
                    if d >= len {
                        out.extend_from_within(st..st + len);
                    } else {
                        for i in 0..len {
                            let c = out[st + i];
                            out.push(c);
                        }
                    }
                    if out.len() > limit {
                        return None;
                    }
                }
            }
            _ => return None,
        }
        if out.len() > limit {
            return None;
        }
        if last == 1 {
            break;
        }
    }
    // unread whole bytes in the bit buffer belong to what follows the stream
    let used = r.pos - (r.n / 8) as usize;
    Some((out, used))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_rebuild_the_stream() {
        // repetitive text with noise: many blocks, long hash chains
        let mut x = 7u64;
        let data: Vec<u8> = (0..3_000_000u32)
            .flat_map(|i| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if x % 5 == 0 { vec![(x >> 8) as u8] } else { format!("w{} ", (i * 7 + (x as u32 % 300)) % 911).into_bytes() }
            })
            .collect();
        for level in [4u8, 6, 9] {
            let full = deflate(&data, level, 8);
            let pts = deflate_points(&data, level, 8, &full).expect("points");
            assert!(pts.len() > 4, "level {level}: {} points", pts.len());
            let cuts: Vec<(usize, u64)> = pts.iter().step_by(pts.len() / 4).copied().collect();
            let mut out = Vec::new();
            let mut len = 0u64;
            let mut start = 0;
            for k in 0..=cuts.len() {
                let stop = cuts.get(k).map(|c| c.0);
                let (seg, bits) = deflate_seg(&data, level, 8, start, stop, (len & 7) as u32).expect("segment");
                if let Some(c) = cuts.get(k) {
                    assert_eq!(len + bits - (len & 7), c.1, "level {level} segment {k} length");
                }
                append_bits(&mut out, &mut len, &seg, bits);
                start = stop.unwrap_or(0);
            }
            assert_eq!(len, full.len() as u64 * 8);
            assert!(out == full, "level {level}: rebuilt stream differs");
        }
    }

    /// Writes NZ_FILE as a gzip member at level 9 to NZ_OUT (test input for rezip):
    /// `NZ_FILE=in NZ_OUT=out.gz cargo test --release write_gzip9 -- --ignored`
    #[test]
    #[ignore]
    fn write_gzip9() {
        let data = std::fs::read(std::env::var("NZ_FILE").unwrap()).unwrap();
        let mut out = vec![0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 2, 3];
        out.extend_from_slice(&deflate(&data, 9, 8));
        out.extend_from_slice(&crate::codec::crc32(&data).to_le_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        std::fs::write(std::env::var("NZ_OUT").unwrap(), out).unwrap();
    }

    #[test]
    fn deflate_inflate_round_trip() {
        let mut x = 7u64;
        let mut data: Vec<u8> = Vec::new();
        for i in 0..300_000u32 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            data.push(if i % 1000 < 600 { b"the quick brown fox jumps "[(i % 26) as usize] } else { (x % 7) as u8 + b'a' });
        }
        for level in 1..=9 {
            for mem in [1, 8, 9] {
                let c = deflate(&data, level, mem);
                let (d, used) = inflate(&c, usize::MAX).unwrap();
                assert_eq!(used, c.len());
                assert!(d == data, "level {level} mem {mem}");
            }
        }
        let c = deflate(&[], 6, 8);
        assert_eq!(inflate(&c, 10).unwrap().0, Vec::<u8>::new());
    }

    /// Deflate speed (and output fingerprints, which must not change) on a file:
    /// `NZ_FILE=... cargo test --release deflate_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn deflate_timing() {
        let data = std::fs::read(std::env::var("NZ_FILE").unwrap()).unwrap();
        for (level, mem) in [(1, 8), (6, 8), (9, 8)] {
            let t = std::time::Instant::now();
            let c = deflate(&data, level, mem);
            let s = t.elapsed().as_secs_f64();
            let h = c.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
            let t = std::time::Instant::now();
            assert!(deflate_eq(&data, level, mem, &c));
            let se = t.elapsed().as_secs_f64();
            eprintln!("level {level}: {} -> {} ({:016x}) deflate {:.1} MB/s, deflate_eq {:.1} MB/s", data.len(), c.len(), h, data.len() as f64 / 1e6 / s, data.len() as f64 / 1e6 / se);
        }
    }
}
