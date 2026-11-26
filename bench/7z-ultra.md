# 7-Zip LZMA2 Ultra reference results

These are fixed reference numbers. nzip is compared against them; 7-Zip is not re-run.

- 7-Zip 26.06 (x64), Linux
- Command: `7z a -t7z -m0=lzma2 -mx=9 -mmt=on out.7z <input>`, extract with `7z x`
- Machine: AMD Ryzen 9 3900X (12 cores / 24 threads), 135 GB RAM, NVMe; files in page cache
- Times are wall clock. Every archive was extracted and compared with `diff -r`.

| Data | Input bytes (sum of files) | 7z bytes | Compress | Decompress | Measured |
|---|---:|---:|---:|---:|---|
| Portal 2 folder (`~/Downloads/Portal 2`, 46 files) | 237,637,050 | 143,808,833 | 73.7 s | 4.5 s | 2026-11-22 |
| TestSet folder (`~/Downloads/TestSet`, 32 files) | 4,372,581 | 3,344,553 | 0.32 s | 0.09 s | 2026-11-22 |
| Firefox folder (`data/bench/FireFox/Mozilla Firefox`, 81 files) | 381,054,262 | 93,338,587 | 78.3 s | 3.03 s | 2026-11-22 |
| enwik8 | 100,000,000 | 24,799,463 | 40.3 s | 0.80 s | 2026-11-22 |
| Silesia folder (`data/bench/silesia`, 12 files) | 211,938,580 | 48,688,243 | 39.4 s | 1.51 s | 2026-11-22 |
| enwik9 | 1,000,000,000 | 210,604,362 | 212.5 s | 3.1 s | earlier session |

## Single files

Times measured 2026-11-25 (same command and machine; each extract compared with `cmp`).

| File | Input bytes | 7z bytes | Compress | Decompress |
|---|---:|---:|---:|---:|
| Portal 2 `host64/nvngx_dlssnr.dll` | 165,830,144 | 117,991,100 | 45.83 s | 3.11 s |
| Portal 2 `host64/nvngx_dlss.dll` | 58,956,912 | 22,320,108 | 7.63 s | 0.70 s |
| Portal 2 `host64/dxgi.dll` | 5,592,064 | 1,655,430 | 0.62 s | 0.06 s |
| Firefox `xul.dll` | 180,460,160 | 49,665,712 | 34.30 s | 1.63 s |
| Firefox `xul.dll`, no exe filter (`-mf=off`) | 180,460,160 | 54,895,306 | 39.97 s | 1.74 s |
| Firefox `omni.ja` | 50,017,943 | 10,584,164 | 9.08 s | 0.34 s |
| Firefox `browser/omni.ja` | 60,032,635 | 11,556,518 | 9.11 s | 0.34 s |
| Firefox `dxcompiler.dll` | 24,224,384 | 5,825,030 | 3.18 s | 0.20 s |
| Firefox `onnxruntime.dll` | 11,937,920 | 2,808,656 | 1.14 s | 0.10 s |
| Firefox `gkcodecs.dll` | 11,016,320 | 2,942,420 | 0.93 s | 0.11 s |

On 2026-11-25 `xul.dll` packed to 49,526,597 bytes (the row above keeps the recorded
49,665,712); every other file matched its recorded size exactly.

## Lodestone samples

`data/bench/lodestone/<type>`: ~1 GB samples per file type of the 107 GB lodestone build
tree (file lists in `<type>.list`). Measured 2026-11-25, same command and machine;
decompress time is `7z t` (test, no files written).

| Type | Input bytes | 7z bytes | Compress | Decompress |
|---|---:|---:|---:|---:|
| ar | 991,263,932 | 109,765,003 | 48.11 s | 1.23 s |
| exe | 1,171,197,296 | 156,142,573 | 46.31 s | 1.56 s |
| exebig | 1,365,310,504 | 173,008,251 | 54.90 s | 3.34 s |
| incr | 999,999,963 | 239,233,439 | 37.69 s | 2.61 s |
| jar | 999,999,518 | 639,777,998 | 31.71 s | 3.76 s |
| media | 999,999,647 | 689,122,664 | 25.18 s | 1.82 s |
| obj | 999,999,856 | 72,980,940 | 41.24 s | 1.05 s |
| rlib | 999,998,988 | 145,732,833 | 44.49 s | 1.75 s |
| rmeta | 999,999,086 | 195,351,213 | 44.82 s | 2.05 s |
| so | 999,995,768 | 105,548,013 | 44.69 s | 1.97 s |
| text | 426,985,170 | 18,417,692 | 28.74 s | 0.56 s |

Whole tree (`~/Downloads/lodestone-minecraft-launcher`, 107.3 GB), measured by the user:
16,437,736,629 bytes in 33.1 min.

## TestSet, per file

| File | Input bytes | 7z bytes |
|---|---:|---:|
| 21-website.png | 1,814,440 | 1,702,600 |
| Lodestone Launcher (standalone).html | 1,772,556 | 1,318,062 |
| composer.lock | 161,643 | 13,755 |
| sync-modal.webp | 157,900 | 158,051 |
| en_us.toml | 86,541 | 20,623 |
| icon.icns | 78,881 | 51,443 |
| AGENTS.md | 61,787 | 23,849 |
| pnpm-lock.yaml | 50,980 | 17,538 |
| update.rs | 39,925 | 8,425 |
| SyncRouteTest.php | 28,052 | 5,249 |
| FriendsPage.tsx | 19,543 | 3,876 |
| technic.ico | 15,086 | 1,804 |
| scene-snow-2.svg | 10,419 | 3,022 |
| record.sh | 9,006 | 3,562 |
