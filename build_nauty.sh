#!/usr/bin/env bash
# Build the vendored nauty static library (nauty/nauty.a) with PORTABLE, PIC flags.
#
# Why this script exists (resolves two pieces of tech debt):
#   1. The Python bridge links nauty/nauty.a statically into a shared library, so nauty
#      MUST be compiled -fPIC or the link fails with a relocation error. This script always
#      passes -fPIC, removing the manual "rebuild nauty by hand first" step on a clean checkout.
#   2. nauty's ./configure autodetects -march=native by default, baking the BUILD host's exact
#      instruction set into the archive. Copying such a build to another machine risks
#      illegal-instruction crashes or quietly mistuned wall-clock numbers.
#      --enable-generic disables -march=native, so the build is portable and reproducible; the
#      intended workflow is to run this script ON the target machine (popcnt etc. are still
#      autodetected correctly for that host).
#
# nauty itself is gitignored (per-machine third-party). If the nauty/ source tree is absent
# this script extracts it from a vendored nauty*.tar.gz next to it (pre-fetch the tarball on a
# login node for offline compute nodes). Downloading is out of scope: https://pallini.di.uniroma1.it/
#
# Idempotent: reconfigures only when the generated makefile is missing or was not produced with
# our PIC flags (detected by grepping the makefile's CFLAGS line); otherwise just runs an
# incremental `make nauty.a`. Set NAUTY_FORCE=1 to force a clean reconfigure+rebuild.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
cd "$root"

want_cflags="-O2 -fPIC"

# --- 0. Obtain the nauty source tree if absent (extract a vendored tarball) ---
if [ ! -d nauty ]; then
  tarball="$(ls -1 nauty*.tar.gz 2>/dev/null | head -1 || true)"
  if [ -z "$tarball" ]; then
    echo "build_nauty.sh: no nauty/ source tree and no nauty*.tar.gz to extract." >&2
    echo "  Fetch nauty from https://pallini.di.uniroma1.it/ (e.g. nauty2_9_3.tar.gz) here first." >&2
    exit 1
  fi
  echo "build_nauty.sh: extracting $tarball -> nauty/"
  tar xzf "$tarball"
  # nauty release tarballs extract to a dir matching the tarball basename (e.g.
  # nauty2_9_3.tar.gz -> nauty2_9_3/); fall back to the sole new nauty* dir otherwise.
  extracted="$(basename "$tarball" .tar.gz)"
  if [ ! -d "$extracted" ]; then
    extracted="$(find . -maxdepth 1 -type d -name 'nauty*' ! -name nauty | head -1)"
  fi
  mv "$extracted" nauty
fi

cd nauty

# --- 1. (Re)configure with portable + PIC flags only when needed ---
needs_configure=0
if [ ! -f makefile ] || ! grep -q -- '-fPIC' makefile; then
  needs_configure=1
fi
if [ "${NAUTY_FORCE:-0}" = "1" ]; then
  needs_configure=1
fi

if [ "$needs_configure" -eq 1 ]; then
  [ -f makefile ] && make clean >/dev/null 2>&1 || true
  ./configure --enable-generic CFLAGS="$want_cflags"
fi

# --- 2. Build the static archive the engine/bridge link against ---
make nauty.a

# Verify, don't assume. `make` can report "nauty.a is up to date" and exit 0 while no
# archive exists -- e.g. after copying the tree with mtimes preserved (rsync -a), which
# leaves make's timestamp reasoning satisfied by prerequisites that were never built.
# Announcing success there sends you off to debug CMake instead of nauty; CMake then
# silently skips the _core extension (it is guarded on nauty.a existing) and the Python
# bridge fails to import with no hint as to why. Re-run with NAUTY_FORCE=1 to recover.
if [ ! -f nauty.a ]; then
  echo "build_nauty.sh: make reported success but $root/nauty/nauty.a does not exist." >&2
  echo "  Stale build state (often mtimes preserved by a copy). Retry: NAUTY_FORCE=1 $0" >&2
  exit 1
fi
echo "build_nauty.sh: built $root/nauty/nauty.a"
