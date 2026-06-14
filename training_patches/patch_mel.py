"""
Workaround for `CUFFT_INTERNAL_ERROR` from torch.stft on some GPU/CUDA
combinations (seen on RTX 4090 with torch 1.13.1+cu117 - cuFFT from CUDA 11.7
doesn't fully support Ada). Moves the STFT computation to CPU and the result
back to the original device.

Only needed on older torch/CUDA builds. Not required with torch 2.7+cu128 on
RTX 5090.

Usage: python3 patch_mel.py /path/to/piper/src/python/piper_train/vits/mel_processing.py
"""
import sys

path = sys.argv[1]
src = open(path).read()

src = src.replace("torch.stft(\n            y,", "torch.stft(\n            y.cpu(),")
src = src.replace(
    "window=hann_window[wnsize_dtype_device],",
    "window=hann_window[wnsize_dtype_device].cpu(),",
)

old_close = "            return_complex=True,\n        )\n    )\n"
new_close = "            return_complex=True,\n        )\n    ).to(y.device)\n"
n = src.count(old_close)
src = src.replace(old_close, new_close)

open(path, "w").write(src)
print("patched occurrences:", n)
