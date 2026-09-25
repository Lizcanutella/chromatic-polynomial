#!/usr/bin/env bash
# Verified build (no cmake required). Requires: g++ >= 13, libgmp-dev (provides gmpxx.h).
#   Ubuntu/Debian:  sudo apt-get install -y g++ libgmp-dev
set -euo pipefail
cd "$(dirname "$0")"
g++ -std=c++17 -O2 -Iinclude tests/validate.cpp -lgmpxx -lgmp -o validate
echo "built ./validate - running:"
./validate
