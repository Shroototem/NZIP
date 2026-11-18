#!/usr/bin/env bash
# nzip self-test: generates varied data, round-trips it through every mode and device,
# and checks the output is byte-identical. Writes only to a temporary directory.
set -u
cd "$(dirname "$0")"
NZ=./nzip
T=$(mktemp -d "${TMPDIR:-/tmp}/nzip-selftest.XXXXXX")
trap 'rm -rf "$T"' EXIT
fail=0

echo "nzip $($NZ --version | awk '{print $2}') self-test on $(uname -srm)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
  echo "GPU: none detected (CPU only)"
fi

# Test inputs: empty, tiny, text, binary (this program), random, a mix with a far repeat,
# and data with gzip streams inside.
: > "$T/empty"
printf 'hello' > "$T/tiny"
for i in $(seq 1 200000); do echo "line $i: the quick brown fox jumps over the lazy dog $((i % 97))"; done > "$T/text"
cp "$NZ" "$T/binary"
head -c 8000000 /dev/urandom > "$T/random"
cat "$T/text" "$T/random" "$T/binary" "$T/text" > "$T/mixed"
# Deflate streams inside other data (unpacked and rebuilt bit-exact).
if command -v gzip >/dev/null 2>&1; then
  { head -c 3000000 "$T/text"; gzip -6 -c "$T/text"; cat "$T/binary"; gzip -1 -c "$T/binary"; } > "$T/deflate"
else
  cp "$T/mixed" "$T/deflate"
fi

for f in empty tiny text binary random mixed deflate; do
  for mode in max fast store; do
    for dev in auto cpu; do
      if ! $NZ -c -q -f -m $mode -D $dev "$T/$f" "$T/$f.nz" 2>"$T/err"; then
        echo "FAIL compress $f $mode $dev: $(cat "$T/err")"; fail=1; continue
      fi
      if ! $NZ -d -q -f -D $dev "$T/$f.nz" "$T/$f.out" 2>"$T/err" || ! cmp -s "$T/$f" "$T/$f.out"; then
        echo "FAIL roundtrip $f $mode $dev"; fail=1; continue
      fi
      $NZ -t "$T/$f.nz" >/dev/null 2>&1 || { echo "FAIL test $f $mode $dev"; fail=1; }
    done
  done
  printf '  %-7s %10s bytes -> %10s bytes (max)  ok\n' "$f" "$(stat -c %s "$T/$f")" \
    "$($NZ -c -q -f "$T/$f" "$T/$f.nz" && stat -c %s "$T/$f.nz")"
done

# Folders: pack, verify, restore, and compare contents, permissions and symlinks.
mkdir -p "$T/dir/sub/deeper" "$T/dir/empty"
cp "$T/text" "$T/dir/sub/deeper/notes.txt"; printf '#!/bin/sh\n' > "$T/dir/run.sh"; chmod 755 "$T/dir/run.sh"
ln -s sub/deeper/notes.txt "$T/dir/link"
if $NZ -c -q -f "$T/dir" "$T/dir.nz" && $NZ -t "$T/dir.nz" >/dev/null && $NZ -d -q "$T/dir.nz" "$T/dir.out" \
   && diff -r --no-dereference "$T/dir" "$T/dir.out" >/dev/null \
   && [ "$(cd "$T/dir" && find . -printf '%m %y %p %l\n' | sort)" = "$(cd "$T/dir.out" && find . -printf '%m %y %p %l\n' | sort)" ]; then
  echo "  folder round trip (contents, permissions, symlinks)  ok"
else
  echo "FAIL folder round trip"; fail=1
fi

# Corruption must be detected, never silently decoded.
$NZ -c -q -f "$T/text" "$T/c.nz"
printf '\xff' | dd of="$T/c.nz" bs=1 seek=5000 conv=notrunc status=none
if $NZ -t "$T/c.nz" >/dev/null 2>&1; then echo "FAIL corruption not detected"; fail=1; else echo "  corrupted archive rejected  ok"; fi

if [ $fail = 0 ]; then echo "ALL PASSED"; else echo "SOME TESTS FAILED"; exit 1; fi
