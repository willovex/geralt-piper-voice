#!/usr/bin/env python3
"""
Etap 4: Scala transkrypcje, resampleuje audio do 22050Hz, przygotowuje surowy dataset Pipera
(format LJSpeech: wavs/*.wav + metadata.csv).

Wyjscie trafia do /data/training_raw/ -- to jest --input-dir dla
`python -m piper_train.preprocess`, ktory dalej generuje binarny dataset
treningowy (config.json, dataset.jsonl, cache/) w /data/training_processed/
(czyli --dataset-dir dla piper_train).
"""
import csv, os, sys
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from math import gcd

DATA = Path('/data')
FILTERED = DATA / 'geralt_filtered'
WAVS_OUT = DATA / 'training_raw' / 'wavs'
META_OUT  = DATA / 'training_raw' / 'metadata.csv'
TARGET_SR = 22050
MIN_WORDS = 2

WAVS_OUT.mkdir(parents=True, exist_ok=True)

print('Scalanie transkrypcji...', flush=True)
rows = []
for i in list(range(3)) + ['all']:
    csv_path = DATA / f'metadata_{i}.csv'
    if not csv_path.exists():
        print(f'  BRAK: {csv_path}', flush=True)
        continue
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='|'):
            if len(row) >= 2 and len(row[1].split()) >= MIN_WORDS:
                rows.append(row)

print(f'Scalono {len(rows)} transkrypcji.', flush=True)

# Deduplikacja po sciezce
seen = set()
unique = []
for r in rows:
    if r[0] not in seen:
        seen.add(r[0])
        unique.append(r)
print(f'Po deduplikacji: {len(unique)}', flush=True)

print('Resampleowanie i zapis do wavs/...', flush=True)
meta_rows = []
errors = 0
for i, (src_path, text) in enumerate(unique):
    src = Path(src_path)
    if not src.exists():
        errors += 1
        continue
    try:
        audio, sr = sf.read(str(src), dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            g = gcd(TARGET_SR, sr)
            audio = resample_poly(audio, TARGET_SR // g, sr // g).astype('float32')
        out_name = src.stem + '.wav'
        out_path = WAVS_OUT / out_name
        sf.write(str(out_path), audio, TARGET_SR, subtype='PCM_16')
        meta_rows.append([f'wavs/{out_name}', 'geralt', text])
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(unique)} ({int(100*(i+1)/len(unique))}%)', flush=True)
    except Exception as e:
        errors += 1

print(f'Bledy: {errors}', flush=True)
print(f'Zapisano {len(meta_rows)} plikow do wavs/', flush=True)

with open(META_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    w.writerows(meta_rows)

print(f'Dataset gotowy -> {META_OUT}', flush=True)
