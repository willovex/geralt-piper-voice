"""
Speed up DataLoaders in piper_train/vits/lightning.py: use many workers with
persistent/pinned memory instead of the default num_workers=1. Needed to keep
a fast GPU (RTX 5090) fed when reading the dataset cache from a network volume.

Usage: python3 patch_dataloader.py /path/to/piper/src/python/piper_train/vits/lightning.py
"""
import sys

path = sys.argv[1]
src = open(path).read()

old = "            num_workers=self.hparams.num_workers,\n            batch_size=self.hparams.batch_size,"
new = (
    "            num_workers=16,\n"
    "            batch_size=self.hparams.batch_size,\n"
    "            persistent_workers=True,\n"
    "            pin_memory=True,"
)

n = src.count(old)
assert n == 3, f"expected 3 dataloader blocks, found {n}"
src = src.replace(old, new)

open(path, "w").write(src)
print(f"patched {n} dataloader(s) in", path)
