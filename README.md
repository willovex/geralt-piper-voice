# Geralt Voice — Piper TTS

Klon głosu Geralta z Rivii (Jacek Rozenek, polski dubbing Wiedźmin 3) jako
lokalny, darmowy głos TTS [Piper](https://github.com/rhasspy/piper),
fine-tuned z bazowego modelu `pl_PL-gosia-medium` (medium quality, 22050 Hz),
zintegrowany z Home Assistant (addon Piper / Wyoming).

Gotowy model: [`model/geralt.onnx`](model/geralt.onnx) +
[`model/geralt.onnx.json`](model/geralt.onnx.json) — checkpoint `epoch=5217`
(fine-tuning od `epoch=5001` modelu bazowego, czyli ~216 dodatkowych epok na
własnym datasecie).

## Demo

Nagranie pokazujące działanie głosu w Home Assistant:

<video src="https://github.com/user-attachments/assets/e96525fb-3a0f-4add-a446-9c926dbc3ba3" controls width="400"></video>

## Zawartość repo

- `model/` — finalny eksport ONNX (Piper) gotowy do wgrania do `/share/piper/` w HA.
- `scripts/` — cały pipeline: ekstrakcja audio, filtrowanie głosu, transkrypcja,
  przygotowanie datasetu, Dockerfile do treningu/eksportu.
- `training_patches/` — patche potrzebne do trenowania `piper_train` na
  nowszych wersjach `torch` / `pytorch-lightning` (2.x) oraz na GPU
  Blackwell (RTX 5090).
- `docs/HISTORY.md` — szczegółowa historia całego projektu (etapy, problemy,
  dwie utraty danych na RunPod i odzyskiwanie, optymalizacje).
- `docs/DEPLOY_HA.md` — jak wgrać/zaktualizować głos w Home Assistant.

## Pipeline (skrót)

| Etap | Opis | Skrypt |
|------|------|--------|
| 1 | Ekstrakcja audio (.wem) z plików gry Wiedźmin 3 | `scripts/w3speech_extractor` (zewnętrzne narzędzie) |
| 2 | Filtrowanie nagrań — wybór tylko głosu Geralta | `scripts/filter_geralt_v4.py` |
| 3 | Transkrypcja (faster-whisper / Whisper) | `scripts/transcribe_parallel.py`, `scripts/transcribe_single.py` |
| 4 | Przygotowanie datasetu LJSpeech (`wavs/` + `metadata.csv`) + `piper_train.preprocess` | `scripts/prepare_dataset.py` |
| 5 | Fine-tuning VITS (piper_train), resume od `pl_PL-gosia-medium` epoch=5001 | `training_patches/setup_rtx5090_pod.sh` |
| 6 | Eksport ONNX + deploy do HA | `scripts/make_onnx_json.py`, `docs/DEPLOY_HA.md` |

Pełny opis i historia w [`docs/HISTORY.md`](docs/HISTORY.md).

## Dataset

- 17 138 wyciętych nagrań głosu Geralta (PCM WAV, mono)
- 16 854 transkrybowanych i użytych w treningu (po filtrowaniu zbyt
  krótkich/pustych)
- Format: LJSpeech (`wavs/*.wav` + `metadata.csv`), 22 050 Hz, single-speaker
- Język: polski (`espeak` voice `pl`)

Dataset (surowe nagrania + cache mel-spektrogramów, ~21GB) nie jest częścią
tego repo — zbyt duży na GitHub.

Pełny checkpoint treningowy `epoch=5217-step=50634.ckpt` (807MB, z optimizer
state, do kontynuacji fine-tuningu) jest dostępny w
[Releases](../../releases/tag/v1.0-epoch5217).

## Trening

- Bazowy checkpoint: `pl_PL-gosia-medium epoch=5001` (rhasspy/piper-checkpoints)
- Fine-tuning do `epoch=5217` (batch size 48–64, precision 32, quality medium)
- GPU: RunPod RTX 4090 → RTX 5090 (po dwóch utratach danych pod, patrz HISTORY)
- Finalny `length_scale: 2.3` w `model/geralt.onnx.json` (Geralt mówi wolniej
  niż domyślny model — lepiej odwzorowuje oryginalny dubbing)

## Kontynuacja treningu

Aby kontynuować fine-tuning od `epoch=5217`:

1. Odtwórz dataset (`training_processed/` + `training_raw/`) — patrz
   `scripts/prepare_dataset.py` i `docs/HISTORY.md`.
2. Na nowym podzie GPU uruchom `training_patches/setup_rtx5090_pod.sh`
   (ustaw `RESUME_CKPT` na `epoch=5217-step=50634.ckpt`).
3. Po każdej epoce wyeksportuj ONNX:
   `python -m piper_train.export_onnx <checkpoint> model/geralt.onnx`
