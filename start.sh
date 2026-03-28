#!/bin/bash
# Find libgomp.so.1 in nix store and expose it to the dynamic linker
# This is needed because LightGBM links against GCC's OpenMP at runtime
GOMP=$(ls /nix/store/*/lib/libgomp.so.1 2>/dev/null | head -1)
if [ -z "$GOMP" ]; then
    GOMP=$(find /nix/store -maxdepth 6 -name libgomp.so.1 2>/dev/null | head -1)
fi
if [ -n "$GOMP" ]; then
    export LD_LIBRARY_PATH="$(dirname "$GOMP"):${LD_LIBRARY_PATH:-}"
fi
cd backend
exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
