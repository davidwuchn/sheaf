#!/usr/bin/env bash
set -euo pipefail

archive="${1:-sheaf-examples.tar.gz}"

bazel build @nanogpt_weights//file
execution_root=$(bazel info execution_root)
weights=$(bazel cquery --output=files @nanogpt_weights//file)

python3 bazel/package_examples.py \
  --archive "$archive" \
  --weights "$execution_root/$weights"

printf 'Created %s\n' "$archive"
