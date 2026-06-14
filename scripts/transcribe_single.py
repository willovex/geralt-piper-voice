#!/usr/bin/env python3
"""
Jeden worker obsługuje wszystkie pliki. Wznawia z metadata_0/1/2.csv.
"""
import csv, os
from pathlib import Path
from faster_whisper import WhisperModel

INPUT_DIR  = Path('/data/geralt_filtered')
OUTPUT_CSV = Path('/data/metadata_all.csv')
MODEL_SIZE = 'medium'

model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8', cpu_threads=6)
print('Model zaladowany.', flush=True)

files = sorted(INPUT_DIR.glob('*.wav'))
total = len(files)
print(f'Znaleziono {total} plikow.', flush=True)

# Zbierz juz zrobione ze wszystkich poprzednich CSV
done = set()
for csv_path in ['/data/metadata_0.csv', '/data/metadata_1.csv', '/data/metadata_2.csv', '/data/metadata_all.csv']:
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='|'):
                if row: done.add(row[0])

remaining = [f for f in files if str(f) not in done]
print(f'Juz zrobione: {len(done)} | Pozostalo: {len(remaining)}', flush=True)

with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter='|', quoting=csv.QUOTE_MINIMAL)
    for i, wav_path in enumerate(remaining):
        try:
            segments, _ = model.transcribe(str(wav_path), language='pl', beam_size=5,
                vad_filter=True, vad_parameters={'min_silence_duration_ms': 300})
            text = ' '.join(seg.text.strip() for seg in segments).strip()
            if text:
                writer.writerow([str(wav_path), text])
                csvfile.flush()
            if (i + 1) % 500 == 0:
                pct = int(100 * (i+1) / len(remaining))
                print(f'  {i+1}/{len(remaining)} ({pct}%)', flush=True)
        except Exception as e:
            print(f'  BLAD {wav_path.name}: {e}', flush=True)

lines = sum(1 for _ in open(OUTPUT_CSV, encoding='utf-8'))
print(f'Gotowe! {lines} transkrypcji w {OUTPUT_CSV}', flush=True)
