#!/usr/bin/env python3
import os, csv, sys
from pathlib import Path
from faster_whisper import WhisperModel

INPUT_DIR = Path('/data/geralt_filtered')
MODEL_SIZE = 'medium'

worker_id  = int(sys.argv[1])   # 0, 1 lub 2
n_workers  = int(sys.argv[2])   # 3
OUTPUT_CSV = Path(f'/data/metadata_{worker_id}.csv')

def main():
    print(f'Worker {worker_id}/{n_workers} startuje...', flush=True)
    model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8', cpu_threads=2)
    print(f'Worker {worker_id}: model załadowany.', flush=True)

    files = sorted(INPUT_DIR.glob('*.wav'))
    my_files = [f for i, f in enumerate(files) if i % n_workers == worker_id]
    print(f'Worker {worker_id}: {len(my_files)} plików do przepisania.', flush=True)

    done = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline='', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='|'):
                if row: done.add(row[0])
        print(f'Worker {worker_id}: wznawiam od {len(done)}.', flush=True)

    remaining = [f for f in my_files if str(f) not in done]

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
                if (i + 1) % 200 == 0:
                    print(f'  W{worker_id}: {i+1}/{len(remaining)} ({int(100*(i+1)/len(remaining))}%)', flush=True)
            except Exception as e:
                print(f'  W{worker_id} BŁĄD {wav_path.name}: {e}', flush=True)

    print(f'Worker {worker_id}: gotowe! {OUTPUT_CSV}', flush=True)

if __name__ == '__main__':
    main()
