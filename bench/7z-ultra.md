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

| File | Input bytes | 7z bytes |
|---|---:|---:|
| Portal 2 `host64/nvngx_dlssnr.dll` | 165,830,144 | 117,991,100 |
| Portal 2 `host64/nvngx_dlss.dll` | 58,956,912 | 22,320,108 |
| Portal 2 `host64/dxgi.dll` | 5,592,064 | 1,655,430 |
| Firefox `xul.dll` | 180,460,160 | 49,665,712 |
| Firefox `xul.dll`, no exe filter (`-mf=off`) | 180,460,160 | 54,895,306 |
| Firefox `omni.ja` | 50,017,943 | 10,584,164 |
| Firefox `browser/omni.ja` | 60,032,635 | 11,556,518 |
| Firefox `dxcompiler.dll` | 24,224,384 | 5,825,030 |
| Firefox `onnxruntime.dll` | 11,937,920 | 2,808,656 |
| Firefox `gkcodecs.dll` | 11,016,320 | 2,942,420 |

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
