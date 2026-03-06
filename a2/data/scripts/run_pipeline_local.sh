#!/usr/bin/env bash
#
# One-click local data pipeline.
# Runs: ingest → preprocess → build → validate
#
# Usage:
#   ./run_pipeline_local.sh          # Full pipeline (all sources)
#   ./run_pipeline_local.sh --mvp    # MVP pipeline (ASL Citizen, MVP vocab only)
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MVP=""
if [[ "$*" == *"--mvp"* ]]; then
  MVP="--mvp"
  echo "=== Running MVP pipeline (ASL Citizen, MVP vocabulary only) ==="
else
  echo "=== Running full pipeline (all sources) ==="
fi

echo ""
echo "Stage 1: Ingest"
python ingest_asl_citizen.py $MVP
if [[ -z "$MVP" ]]; then
  python ingest_wlasl.py
  python ingest_msasl.py
fi

echo ""
echo "Stage 2: Preprocess"
if [[ -n "$MVP" ]]; then
  python preprocess_clips.py --source asl_citizen --mvp
else
  python preprocess_clips.py
fi

echo ""
echo "Stage 3: Build dataset"
python build_unified_dataset.py $MVP

echo ""
echo "Stage 4: Validate"
python validate.py $MVP

echo ""
echo "=== Pipeline complete ==="
