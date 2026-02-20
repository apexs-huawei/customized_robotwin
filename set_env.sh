#!/usr/bin/env bash

# absolute path to robotwin_bench
export ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# workspace root (parent of robotwin_bench)
export WORKSPACE_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"

# robotwin root
export BENCH_ROOT="$WORKSPACE_ROOT/benchmark"

echo "BENCH_ROOT=$BENCH_ROOT"
echo "ROBOTWIN_ROOT=$ROBOTWIN_ROOT"
