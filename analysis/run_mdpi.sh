#!/usr/bin/env bash
# Runs Studies A-C of the MDPI redundancy article in order, one log per step.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs
for s in A1_selection_synthetic A2_analyze_synthetic B1_extract_variants B2_measurement_properties C1_clinical_severity; do
  echo "[$(date +%T)] $s"
  python "$s.py" > "../logs/mdpi_$s.txt" 2>&1 || { echo "FAILED: see logs/mdpi_$s.txt"; exit 1; }
done
echo "done: results in ../output"
