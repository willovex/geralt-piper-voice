#!/usr/bin/env python3
"""
Filtruje głos Geralta używając torchaudio MFCC + delta + delta-delta.
Zdecydowanie lepsza dyskryminacja niż ręczny MFCC z v3.
"""
import os
import subprocess
import tempfile
import shutil
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from scipy.signal import correlate
from multiprocessing import Pool
import random

INPUT_DIR  = Path("/Users/piotrgarbacz/Documents/geralt_voice/wem_all")
OUTPUT_DIR = Path("/Users/piotrgarbacz/Documents/geralt_voice/geralt_filtered")
REFS       = [
    "/Users/piotrgarbacz/Documents/geralt_voice/próbki_do_odsłuchu/0x000fe3d6.wav",
    "/Users/piotrgarbacz/Documents/geralt_voice/próbki_do_odsłuchu/0x000ffb4d.wav",
]
VGMSTREAM     = "/usr/local/bin/vgmstream-cli"
MIN_DUR       = 1.0
MAX_DUR       = 15.0
MIN_PITCH     = 80
MAX_PITCH     = 130
SIM_THRESHOLD = 0.97
TARGET_SR     = 16000
OUTPUT_DIR.mkdir(exist_ok=True)


def get_embedding(wav_path):
    """Ładuje WAV z dysku i oblicza embedding MFCC+delta+delta2."""
    waveform, sr = torchaudio.load(str(wav_path))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR

    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=40,
        melkwargs={"n_fft": 512, "hop_length": 160, "n_mels": 80, "f_min": 50.0},
    )
    mfcc = mfcc_transform(waveform)       # (1, 40, T)
    delta  = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mfcc, delta, delta2], dim=1)  # (1, 120, T)
    embedding = features.mean(dim=2).squeeze(0)          # (120,)
    norm = embedding.norm()
    return (embedding / norm).numpy() if norm > 0 else embedding.numpy()


def estimate_pitch(audio, sr):
    audio = audio - np.mean(audio)
    frame = int(0.04 * sr)
    min_lag = max(1, int(sr / MAX_PITCH))
    max_lag = int(sr / MIN_PITCH)
    pitches = []
    for start in range(0, len(audio) - frame, frame // 2):
        chunk = audio[start:start + frame]
        if np.sqrt(np.mean(chunk**2)) < 0.005:
            continue
        corr = correlate(chunk, chunk, mode='full')
        corr = corr[len(corr)//2:]
        if max_lag >= len(corr):
            continue
        window = corr[min_lag:max_lag]
        if not len(window):
            continue
        peak = np.argmax(window) + min_lag
        if corr[peak] > 0.25 * corr[0]:
            pitches.append(sr / peak)
    return float(np.median(pitches)) if pitches else 0.0


def build_reference(ref_paths):
    vecs = [get_embedding(p) for p in ref_paths]
    avg = np.mean(vecs, axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def process_file(args):
    wav_path, ref_vec, threshold = args
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        r = subprocess.run([VGMSTREAM, '-o', tmp_path, str(wav_path)],
                           capture_output=True, timeout=10)
        if r.returncode != 0:
            return None

        info = sf.info(tmp_path)
        if not (MIN_DUR <= info.duration <= MAX_DUR):
            return None

        audio, sr = sf.read(tmp_path, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        pitch = estimate_pitch(audio, sr)
        if not (MIN_PITCH <= pitch <= MAX_PITCH):
            return None

        vec = get_embedding(tmp_path)
        sim = float(np.dot(vec, ref_vec))

        if sim >= threshold:
            dest = OUTPUT_DIR / wav_path.name
            shutil.copy2(tmp_path, dest)
            return (wav_path.name, round(sim, 3))
        return None
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def calibrate_threshold(ref_vec, sample_files, n=50):
    """Oblicza podobieństwa dla próbki i sugeruje próg."""
    sims = []
    for f in sample_files[:n]:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            r = subprocess.run([VGMSTREAM, '-o', tmp_path, str(f)],
                               capture_output=True, timeout=10)
            if r.returncode == 0:
                info = sf.info(tmp_path)
                if MIN_DUR <= info.duration <= MAX_DUR:
                    audio, sr = sf.read(tmp_path, dtype='float32')
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    pitch = estimate_pitch(audio, sr)
                    if MIN_PITCH <= pitch <= MAX_PITCH:
                        vec = get_embedding(tmp_path)
                        sim = float(np.dot(vec, ref_vec))
                        sims.append((f.name, round(sim, 3), round(pitch, 1)))
        except Exception:
            pass
        finally:
            try: os.unlink(tmp_path)
            except: pass

    return sims


def main():
    print("Buduję referencję głosu Geralta (torchaudio MFCC+delta+delta2)...", flush=True)
    ref_vec = build_reference(REFS)
    print(f"Embedding dim: {ref_vec.shape[0]}", flush=True)

    # Test podobieństwa samych referencji
    sim_rr = float(np.dot(get_embedding(REFS[0]), get_embedding(REFS[1])))
    print(f"Podobieństwo ref0↔ref1 (Geralt vs Geralt): {sim_rr:.3f}", flush=True)

    # Kalibracja — 50 losowych plików
    files = list(INPUT_DIR.rglob("*.wav"))
    total = len(files)
    sample = random.sample(files, min(100, total))

    print(f"\nKalibracja na {min(100, total)} losowych plikach...", flush=True)
    sims = calibrate_threshold(ref_vec, sample)
    if sims:
        all_sims = [s[1] for s in sims]
        print(f"  Próg: {SIM_THRESHOLD} | Zakres podobieństw w próbce: {min(all_sims):.3f}–{max(all_sims):.3f}", flush=True)
        print(f"  Top-5 podobnych plików w próbce (mogą być Geralt):", flush=True)
        for name, sim, pitch in sorted(sims, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {name}: sim={sim}, pitch={pitch}Hz", flush=True)
        retained_pct = sum(1 for s in all_sims if s >= SIM_THRESHOLD) / len(all_sims) * 100
        print(f"  Szacowany % zachowanych (po filtrze pitch+sim): {retained_pct:.1f}%", flush=True)

    print(f"\nFiltruję {total} plików (6 procesów)...", flush=True)
    kept = 0
    args_list = [(f, ref_vec, SIM_THRESHOLD) for f in files]
    with Pool(processes=6) as pool:
        for i, result in enumerate(pool.imap_unordered(process_file, args_list, chunksize=10)):
            if result:
                kept += 1
                if kept <= 10:
                    print(f"  ✓ {result[0]} (sim={result[1]})", flush=True)
            if (i + 1) % 2000 == 0:
                pct = int(100 * kept / (i+1))
                print(f"  {i+1}/{total} | zachowanych: {kept} ({pct}%)", flush=True)

    print(f"\nGotowe! Zachowano {kept} plików Geralta w {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
