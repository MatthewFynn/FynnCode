#!/usr/bin/env bash
set -euo pipefail

# Absolute or relative paths to your scripts
SCRIPTS=(
  "/home/sparc/dev/code/run_multiple_cmdarg_SingleChan_UNET1D.sh"
  "/home/sparc/dev/code/run_multiple_cmdarg_SingleChan_UNET_MFCC.sh"
  "/home/sparc/dev/code/run_multiple_cmdarg_SingleChan_ENC.sh"
  "/home/sparc/dev/code/run_multiple_cmdarg_SingleChan_opera.sh"
)

for s in "${SCRIPTS[@]}"; do
  echo "=================================================="
  echo "Starting: $s  at $(date '+%F %T')"
  echo "=================================================="

  # Run the script; if it exits non-zero, stop the chain
  if ! bash "$s"; then
    echo "ERROR: $s failed (exit code $?). Aborting chain."
    exit 1
  fi

  echo "--------------------------------------------------"
  echo "Finished: $s  at $(date '+%F %T')"
  echo "--------------------------------------------------"
  echo
done

echo "🎉 All scripts completed successfully."
