#!/bin/bash
# Geralt Voice Pipeline — autopilot
# Działa w tle na Unraid, przechodzi przez wszystkie etapy automatycznie.
# Uruchom: nohup bash /mnt/user/geralt_voice/autopilot.sh &

DATA=/mnt/user/geralt_voice
LOG=$DATA/pipeline_log.txt

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }

log '=== AUTOPILOT START ==='

# ── ETAP 3: Czekaj na koniec transkrypcji ──────────────────────────────────
log 'Etap 3: Monitoruję transkrypcję...'
while true; do
  TOTAL=$(cat $DATA/metadata_all.csv $DATA/metadata_0.csv $DATA/metadata_1.csv $DATA/metadata_2.csv 2>/dev/null | wc -l)
  W0=$(docker inspect -f '{{.State.Running}}' geralt_w0 2>/dev/null || echo false)
  W1=$(docker inspect -f '{{.State.Running}}' geralt_w1 2>/dev/null || echo false)
  W2=$(docker inspect -f '{{.State.Running}}' geralt_w2 2>/dev/null || echo false)
  log "Transkrypcja: $TOTAL linii | workery: w0=$W0 w1=$W1 w2=$W2"

  # Gotowe gdy wszystkie workery stopped AND mamy min 15000 linii
  if [ "$W0" = 'false' ] && [ "$W1" = 'false' ] && [ "$W2" = 'false' ] && [ "$TOTAL" -ge 15000 ]; then
    log 'Transkrypcja ZAKONCZONA. Laczna liczba linii: '$TOTAL
    break
  fi
  # Lub gdy mamy 17000+ (w trakcie działania workerów)
  if [ "$TOTAL" -ge 17000 ]; then
    log 'Transkrypcja ZAKONCZONA (17000+ linii). Zatrzymuję workery.'
    docker stop geralt_w0 geralt_w1 geralt_w2 2>/dev/null
    break
  fi
  sleep 300  # check co 5 minut
done

echo 'STAGE=dataset' > $DATA/pipeline_status.txt

# ── ETAP 4a: Przygotowanie surowego datasetu (LJSpeech: wavs/ + metadata.csv) ──
log 'Etap 4a: Przygotowanie surowego datasetu (training_raw)...'
docker run --rm --name geralt_dataset \
  --cpus=4 --memory=4g \
  -v $DATA:/data \
  python:3.11-slim \
  bash -c 'pip install soundfile scipy numpy -q && python /data/prepare_dataset.py' 2>&1 | tee -a $LOG

if [ ! -f $DATA/training_raw/metadata.csv ]; then
  log 'BLAD: prepare_dataset nie stworzyl training_raw/metadata.csv!'
  exit 1
fi
RAW_LINES=$(wc -l < $DATA/training_raw/metadata.csv)
log "Surowy dataset gotowy: $RAW_LINES linii w training_raw/metadata.csv"

# ── ETAP 4b: Build obrazu Docker z piper_train (espeak-ng + monotonic_align) ──
log 'Etap 4b: Budowanie obrazu geralt_piper (piper_train + espeak-ng)...'
docker build -t geralt_piper:latest -f $DATA/Dockerfile.piper $DATA 2>&1 | tee -a $LOG

if ! docker image inspect geralt_piper:latest >/dev/null 2>&1; then
  log 'BLAD: budowanie obrazu geralt_piper:latest nie powiodlo sie!'
  exit 1
fi
log 'Obraz geralt_piper:latest gotowy.'

# ── ETAP 4c: piper_train.preprocess -> training_processed/ (config.json, dataset.jsonl) ──
log 'Etap 4c: Preprocessing datasetu (piper_train.preprocess, jezyk pl)...'
rm -rf $DATA/training_processed
docker run --rm --name geralt_preprocess \
  --cpus=4 --memory=4g \
  -v $DATA:/data \
  geralt_piper:latest \
  python -m piper_train.preprocess \
    --language pl \
    --input-dir /data/training_raw \
    --output-dir /data/training_processed \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050 2>&1 | tee -a $LOG

if [ ! -f $DATA/training_processed/dataset.jsonl ]; then
  log 'BLAD: piper_train.preprocess nie stworzyl training_processed/dataset.jsonl!'
  exit 1
fi
PROC_LINES=$(wc -l < $DATA/training_processed/dataset.jsonl)
log "Dataset przetworzony: $PROC_LINES utterances w training_processed/dataset.jsonl"
echo 'STAGE=training' > $DATA/pipeline_status.txt

# ── ETAP 5a: Pobierz checkpoint bazowy pl_PL-gosia-medium (do fine-tune) ──────
log 'Etap 5a: Pobieranie checkpointu bazowego pl_PL-gosia-medium (846MB)...'
mkdir -p $DATA/training
if [ ! -s $DATA/training/base_model.ckpt ]; then
  docker run --rm --cpus=1 --memory=512m -v $DATA/training:/out alpine:latest \
    sh -c 'apk add --no-cache curl >/dev/null && curl -L -A "Mozilla/5.0" -o /out/base_model.ckpt "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/pl/pl_PL/gosia/medium/epoch%3D5001-step%3D1457672.ckpt?download=true"' 2>&1 | tee -a $LOG
  log 'Checkpoint bazowy pobrany.'
else
  log 'Checkpoint bazowy juz istnieje, pomijam pobieranie.'
fi

if [ ! -s $DATA/training/base_model.ckpt ]; then
  log 'BLAD: brak base_model.ckpt po probie pobrania!'
  exit 1
fi

# ── ETAP 5b: Trening ──────────────────────────────────────────────────────
# Uwaga: checkpoint bazowy ma current_epoch=5001, wiec --max_epochs musi byc > 5001.
# checkpoint-epochs=1 + lightning ModelCheckpoint domyslnie trzyma tylko najnowszy
# ("last.ckpt" / "epoch=N-step=M.ckpt" nadpisywane), wiec zuzycie dysku pozostaje rozsadne.
log 'Etap 5b: Uruchamiam trening Piper (kontener geralt_training)...'
docker rm -f geralt_training 2>/dev/null

docker run -d \
  --name geralt_training \
  --restart=unless-stopped \
  --cpus=4 --memory=7g \
  -v $DATA:/data \
  geralt_piper:latest \
  bash -c '
    echo "[TRAIN] Start treningu piper_train..."
    python -m piper_train \
      --dataset-dir /data/training_processed \
      --accelerator cpu --devices 1 \
      --batch-size 4 \
      --validation-split 0.01 \
      --num-test-examples 0 \
      --max_epochs 8000 \
      --resume_from_checkpoint /data/training/base_model.ckpt \
      --checkpoint-epochs 1 \
      --precision 32 \
      --quality medium

    echo "[TRAIN] Trening zakonczony lub zatrzymany. Eksport finalnego modelu do ONNX..."
    LAST_CKPT=$(find /data/training_processed/lightning_logs -name "*.ckpt" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-)
    if [ -n "$LAST_CKPT" ]; then
      echo "[TRAIN] Uzywam checkpointu: $LAST_CKPT"
      python -m piper_train.export_onnx --checkpoint "$LAST_CKPT" --output-dir /data/training/
      echo "STAGE=done" > /data/pipeline_status.txt
      echo "[TRAIN] GOTOWE! Model ONNX w /data/training/"
    else
      echo "[TRAIN] BLAD: nie znaleziono zadnego checkpointu do eksportu."
    fi
  ' 2>&1 | tee -a $LOG

log 'Kontener geralt_training uruchomiony — trening idzie w tle (restart=unless-stopped).'
log 'Autopilot zakonczyl konfiguracje. Trening bedzie trwal kilka dni.'
echo 'STAGE=training_running' > $DATA/pipeline_status.txt
