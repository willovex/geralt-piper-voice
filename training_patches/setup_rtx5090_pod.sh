#!/bin/bash
# Full from-scratch setup for fine-tuning Piper VITS on a RunPod RTX 5090 pod
# (Blackwell, sm_120). Run on a fresh pod after cloning this repo.
#
# Expects the dataset already present at $DATA_DIR (training_processed/ +
# training_raw/, produced by prepare_dataset.py + piper_train.preprocess)
# and a resume checkpoint at $RESUME_CKPT.
set -euo pipefail

PIPER_DIR=${PIPER_DIR:-/workspace/piper}
DATA_DIR=${DATA_DIR:-/workspace/data}
RESUME_CKPT=${RESUME_CKPT:-/workspace/ckpt/last.ckpt}
PATCH_DIR=$(cd "$(dirname "$0")" && pwd)

echo "== System deps =="
apt-get update -qq
apt-get install -y -qq espeak-ng git rsync build-essential

echo "== torch 2.7.0+cu128 (required for RTX 5090 / sm_120) =="
pip install -q -U typing_extensions
pip install --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# torchvision/torchaudio pinned to cu124 break the pytorch_lightning import (circular import)
pip uninstall -y -q torchvision torchaudio

echo "== piper_train + pytorch-lightning 2.x =="
if [ ! -d "$PIPER_DIR" ]; then
  git clone https://github.com/rhasspy/piper.git "$PIPER_DIR"
fi
pip install -q "pytorch-lightning==2.1.4" "torchmetrics==1.2.1"

echo "== Apply PL2.x / torch2.7 compatibility patches =="
python3 "$PATCH_DIR/patch_main.py" "$PIPER_DIR/src/python/piper_train/__main__.py"
python3 "$PATCH_DIR/patch_lightning.py" "$PIPER_DIR/src/python/piper_train/vits/lightning.py"
python3 "$PATCH_DIR/patch_dataloader.py" "$PIPER_DIR/src/python/piper_train/vits/lightning.py"

# torch 2.6+ defaults torch.load(weights_only=True), which can't unpickle the
# PosixPath objects in older PL 1.7.7 checkpoints.
CLOUD_IO=$(python3 -c "import lightning_fabric, os; print(os.path.join(os.path.dirname(lightning_fabric.__file__), 'utilities', 'cloud_io.py'))")
sed -i 's/torch.load(f, map_location=map_location)/torch.load(f, map_location=map_location, weights_only=False)/' "$CLOUD_IO"

echo "== Build monotonic_align extension =="
cd "$PIPER_DIR/src/python"
python3 piper_train/vits/monotonic_align/setup.py build_ext --inplace
mkdir -p piper_train/vits/monotonic_align/monotonic_align
touch piper_train/vits/monotonic_align/monotonic_align/__init__.py
cp piper_train/vits/monotonic_align/*.so piper_train/vits/monotonic_align/monotonic_align/

echo "== Link /data -> dataset dir (dataset.jsonl uses absolute /data/... paths) =="
ln -sfn "$DATA_DIR" /data

echo "== Done. Start training with: =="
cat <<EOF
cd $PIPER_DIR/src/python
python3 -m piper_train \\
  --dataset-dir $DATA_DIR/training_processed \\
  --accelerator gpu --devices 1 \\
  --batch-size 64 \\
  --validation-split 0.01 --num-test-examples 0 \\
  --max_epochs 8000 \\
  --resume_from_checkpoint $RESUME_CKPT \\
  --checkpoint-epochs 1 \\
  --precision 32 --quality medium
EOF
