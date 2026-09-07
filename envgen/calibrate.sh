#!/usr/bin/env bash
# Run a calibration with the SAME library versions the sandbox image ships, on any machine.
#
#   ./envgen/calibrate.sh fraud_detection            # full report (writes calibration_summary.json)
#   ./envgen/calibrate.sh demand_forecasting --sweep 0.12,0.16,0.22
#   ./envgen/calibrate.sh rating_prediction
#
# Free: CPU only, no API calls.
#
# Why this wrapper exists: xgboost on macOS/arm64 needs an arm64 libomp, and Homebrew here
# is the Intel build (its libomp is x86_64 and cannot load into arm64 Python). Rather than
# install a second Homebrew, this borrows the arm64 libomp that the torch wheel already
# bundles and points dyld at it. On Linux (and inside the sandbox image) the extra bits are
# harmless no-ops. Without this, a calibration silently falls back to sklearn's booster
# (or off-pin library versions) and marks the result "definitive": false.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENTS="$(dirname "$HERE")"
MATS="$(dirname "$ENVIRONMENTS")"

MEMBER="${1:-}"
case "$MEMBER" in
  fraud_detection)     SCRIPT="calibrate_fraud.py" ;;
  demand_forecasting)  SCRIPT="calibrate_demand_forecasting.py" ;;
  rating_prediction)   SCRIPT="calibrate_rating_prediction.py" ;;
  *)
    echo "usage: ./envgen/calibrate.sh <fraud_detection|demand_forecasting|rating_prediction> [--sweep a,b,c]" >&2
    exit 2 ;;
esac
shift

# Pinned to match environments/sandbox/ml/Dockerfile.
PINS=(--with 'xgboost==2.1.1' --with 'scikit-learn==1.5.2' --with 'numpy==2.1.3' --with torch)

cd "$MATS"
OMP_DIR="$(uv run --with torch python -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib")' | tail -1)"
echo "using libomp from: $OMP_DIR"
DYLD_FALLBACK_LIBRARY_PATH="$OMP_DIR" LD_LIBRARY_PATH="${OMP_DIR}:${LD_LIBRARY_PATH:-}" \
  uv run "${PINS[@]}" python "$ENVIRONMENTS/envgen/$SCRIPT" "$@"
