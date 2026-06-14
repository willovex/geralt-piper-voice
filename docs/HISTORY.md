# Historia projektu

## Cel

Sklonować głos Geralta z Rivii (Jacek Rozenek, polski dubbing Wiedźmin 3) jako
lokalny, w pełni darmowy i offline głos TTS dla Home Assistant, w oparciu o
[Piper](https://github.com/rhasspy/piper) (VITS, fine-tuning z
`pl_PL-gosia-medium`).

## Etap 1 — Ekstrakcja audio z gry

Z plików gry Wiedźmin 3 (`.wem`, Wwise audio) wyekstraktowano wszystkie
nagrania dialogowe (`wem_all/`, ~43 465 plików) przy użyciu narzędzi do
ekstrakcji Wwise/REDengine.

## Etap 2 — Filtrowanie głosu Geralta

Ze wszystkich nagrań wybrano tylko te, w których mówi Geralt (rozpoznawanie
głosu/diaryzacja + ręczna/heurystyczna weryfikacja próbek —
`próbki_do_odsłuchu/`, `próbki_weryfikacja/`). Wynik: **17 138 plików PCM WAV**
w `geralt_filtered/`. Skrypt: `scripts/filter_geralt_v4.py` (czwarta iteracja
— wcześniejsze v1-v3 dawały zbyt dużo szumu/innych głosów).

## Etap 3 — Transkrypcja

Transkrypcja 17 138 plików przy użyciu Whisper/faster-whisper, rozdzielona na
3 równoległe workery Docker na serwerze Unraid (`geralt_w0/w1/w2`, po
5713 plików każdy, `--cpus=2 --memory=3g`), zapisujące do `metadata_0/1/2.csv`
+ `metadata_all.csv`. Transkrypcja trwała ponad dobę (~11.5 plików/min na
kontener). Zakończona: **17 124/17 138** transkrybowanych.

Problem: stary proces `autopilot.sh` (uruchomiony przed poprawką) liczył
postęp tylko z `metadata_0/1/2.csv` (453 linie, stała wartość) i nigdy nie
wykryłby końca — wymagał kill+restart po edycji skryptu (edycja pliku nie
wpływa na już działający proces).

## Etap 4 — Przygotowanie datasetu

`scripts/prepare_dataset.py`:
1. Scala `metadata_0/1/2.csv` **i** `metadata_all.csv` (gdzie ląduje większość
   transkrypcji z single-workera) → `metadata_merged.csv`.
2. Filtruje puste linie i zbyt krótkie teksty (<3 słów).
3. Resampling do 22 050 Hz → `training_raw/wavs/`.
4. Tworzy `training_raw/metadata.csv` w formacie Piper
   (`audio/plik.wav|geralt|tekst`).

Następnie `python -m piper_train.preprocess --language pl --input-dir
training_raw --output-dir training_processed --dataset-format ljspeech
--single-speaker --sample-rate 22050` tworzy:
- `training_processed/dataset.jsonl` (16 854 wpisów, ścieżki audio
  `/data/training_raw/wavs/...`)
- `training_processed/cache/22050/*.pt` (33 708 plików — mel-spektrogramy +
  tensory tekstu, po 2 na utterance)
- `training_processed/config.json`

`training_raw/` (2.9GB) i `training_processed/` (18GB) muszą leżeć jako
**siblingi pod jednym katalogiem** zlinkowanym jako `/data` — `dataset.jsonl`
referuje absolutne ścieżki `/data/training_raw/wavs/...`.

Obraz Docker `geralt_piper:latest` (`scripts/Dockerfile.piper`, 4.74GB):
`python:3.10-slim` + `espeak-ng` + `piper` (`pip install -e .`) +
`build_monotonic_align.sh`.

## Etap 5 — Fine-tuning (najbardziej burzliwa część)

### CPU trening na Unraid (równolegle do GPU)

`geralt_training` (Docker, `--cpus=6 --memory=8g→11g`, batch-size 4),
resume od `pl_PL-gosia-medium epoch=5001-step=1457672.ckpt` (846MB,
HuggingFace `rhasspy/piper-checkpoints`, wymaga `curl` z User-Agent i
`?download=true`, inaczej 401).

- ~19.6s/krok, ~4170 kroków/epokę (batch=4) → **~22-23h na epokę** na CPU.
- **OOM kill** tuż przed końcem epoki 1 (zapis checkpointu ~850MB wymaga
  dodatkowego RAM) — naprawione `docker update --memory=11g` (live, bez
  restartu; ale nietrwałe — po restarcie kontenera limit wraca do wartości z
  `docker run`).
- Wyprodukował `epoch=5002-step=1466016.ckpt` — kluczowy w odzyskiwaniu po
  utracie danych na RunPod (patrz niżej).

### GPU trening #1 na RunPod (RTX 4090) — UTRATA DANYCH #1

Równoległy trening na RunPod RTX 4090 doszedł do **epoch=5126**
(`geralt_v2.onnx`, eksport tylko generatora, wdrożony do HA). Następnie
**RunPod "Automatic migration"** (po awarii hosta GPU) zmigrował config poda,
ale **NIE** ephemeral container disk — wszystkie `.ckpt`, `/root/data`,
`/root/piper` przepadły. Przetrwał tylko `geralt_v2.onnx` (inference-only).

**Odzyskanie — opcja A (zrealizowana)**: równoległy CPU-trening na Unraid
wyprodukował `epoch=5002-step=1466016.ckpt` — tylko 124 epoki za utraconym
5126, ale kompletny (z optimizer state).

**Opcja B (przeanalizowana, NIE wykonana)**: wstrzyknięcie wag generatora z
`geralt_v2.onnx` (350 initializerów: `dec.*`, `enc_p.*`, `flow.*`, `dp.*`) do
`epoch=5002.ckpt` (mapowanie nazw `model_g.X` ↔ `X`, `weight_norm`
`weight_g`+`weight_v` ↔ scalony ONNX `weight`, matematycznie odwracalne:
`weight_v=weight`, `weight_g=norm(weight)`). Odrzucone — ryzyko
zdestabilizowania balansu GAN (świeży generator + stary dyskryminator/
optymizer) większe niż koszt 124 epok. Przy budżecie $8 priorytet: stabilny
resume.

### GPU trening #2 na RunPod (RTX 4090, nowy pod `zfgm4moek1j0k6`)

Setup od zera (obraz `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
torch 1.13.1+cu117 wymuszony przez `piper-train~=1.7.0` + `pip<24.1`):

- `pip install "numpy<2" "torchmetrics==0.11.4"` (konflikt torchmetrics 1.5.x
  vs PL 1.7.7).
- Python 3.11 + torch 1.13.1: `field(default=TensorProperties())` w
  `torch/distributed/_shard/sharded_tensor/metadata.py` → mutable default
  error → `sed` na `field(default_factory=TensorProperties)`.
- **RTX 4090 + torch 1.13.1+cu117 → `CUFFT_INTERNAL_ERROR`** w `torch.stft`
  (cuFFT z CUDA 11.7 źle wspiera Ada/sm_89). Fix: `training_patches/patch_mel.py`
  — STFT na CPU, wynik z powrotem na GPU.
- Transfer datasetu RunPod ↔ Unraid: `runpodctl send/receive` (croc). Uwaga:
  `runpodctl send <folder>` najpierw zipuje — jeśli `.zip` już istnieje z
  wcześniejszej próby, wypisuje "file already exists!" i **nie wystawia
  roomu** (proces wisi bez kodu). Fix: `runpodctl send <istniejący>.zip`.
  Transfer 18GB zip ~50min przy ~6MB/s.
- Trening wznowiony od `epoch=5002` (2026-06-13 14:03), GPU 99%,
  ~15GB/24GB VRAM. **Standing requirement**: backup każdego nowego
  checkpointu (epoch>5002) na Unraid przez `runpodctl send/receive`.

### GPU trening #3 — UTRATA DANYCH #2 i pod RTX 5090

Pod `zfgm4moek1j0k6` znów wyparowany (RunPod usunął wszystko, drugi raz —
"Permission denied (publickey)"). Punkt odzyskania: `epoch=5120-step=1548144.ckpt`
(846MB) zabezpieczony wcześniej na Unraid.

Nowy pod: **RTX 5090** (32GB VRAM, Blackwell sm_120, RTX 4090 niedostępna),
obraz `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, Network
Volume (persistent `/workspace`, mfs, 2.1PB).

Setup (patrz `training_patches/setup_rtx5090_pod.sh`):
1. **torch 2.4.1+cu124 (domyślny) nie wspiera sm_120** ("no kernel image is
   available") → upgrade do **torch 2.7.0+cu128** (bez `--no-deps`, inaczej
   `libcusparseLt.so.0` missing). torchvision/torchaudio (cu124) → usunąć
   (circular import `AttributeError: partially initialized module
   'torchvision'`).
2. `pytorch-lightning==2.1.4` + `torchmetrics==1.2.1`.
3. **PL 2.x breaking changes** → `training_patches/patch_main.py`,
   `patch_lightning.py` (manual optimization dla VITS GAN — PL 2.x usunęło
   `optimizer_idx`), `patch_dataloader.py` (num_workers 1→16,
   `persistent_workers`, `pin_memory` — pod ma 32 vCPU / 187GB RAM).
4. **torch 2.6+ `weights_only=True` default** łamie `torch.load` starych
   checkpointów PL 1.7.7 (`PosixPath was not an allowed global`) → sed-patch
   `lightning_fabric/utilities/cloud_io.py` (`weights_only=False`).
5. `monotonic_align`: build z `piper_train/vits/monotonic_align/setup.py
   build_ext --inplace` **z katalogu `src/python`** (nie z wewnątrz
   `monotonic_align/`), `.so` skopiowane do zagnieżdżonego
   `monotonic_align/monotonic_align/` z pustym `__init__.py`
   (`piper_train.vits.monotonic_align.__init__` robi
   `from .monotonic_align.core import maximum_path_c`).
6. Dataset (18GB `training_processed` + 2.9GB `training_raw`, sibling dirs)
   + checkpoint `epoch=5120` rsync'owane bezpośrednio Unraid→pod (klucz
   skopiowany na Unraid jako `/root/.ssh/runpod_geralt_pod`, `sshpass` +
   `-o PreferredAuthentications=password` — pubkey auth do Unraid failuje
   "Too many authentication failures"). `/data → /workspace/data` symlink.

**Trening wznowiony od epoch=5120, batch=64.**

#### I/O bottleneck na sieciowym wolumenie

Mimo patchy, GPU util ~0-18%, trening w stanie "D" (disk-wait):
`/workspace` (Network Volume, mfs) ma zbyt dużą latencję dla wielu małych
plików `.pt` z `num_workers=1`. Próba lokalnej kopii (21GB) na
`/root/data_local` przez `cp -r` była zbyt wolna (~10MB/s sekwencyjnie,
~35min) — przerwana.

**Fix #1**: `patch_dataloader.py` (num_workers 16, `persistent_workers`,
`pin_memory`) + `torch.set_float32_matmul_precision('high')` → GPU 89-91%,
ale tempo nadal niskie (~1 epoka/h przy batch=64, vs ~16/h na RTX4090
batch=48 — niespodziewany 14x spadek).

**Fix #2 (decydujący)**: **równoległa** kopia datasetu na lokalny dysk —
`find ... | xargs -P 32 -I{} cp {} ...` (32 współbieżne procesy `cp`,
ukrywające latencję sieci) zamiast sekwencyjnego `cp -r`. 16 854 plików wav +
33 708 plików cache skopiowane w ~5 minut (vs ~35min sekwencyjnie). Po
restarcie treningu z `/root/data_local`: **GPU 93-99%, VRAM 29.7/32GB,
~11-21 epok/h** (vs ~1/h).

Restart kosztował powrót z `epoch=5121`→`5120` (resume tylko od ostatniego
zapisanego checkpointu), ale netto: **epoch=5120 → epoch=5217** (+97 epok) w
trakcie tej sesji.

#### Cykl backupu (co checkpoint / ~30 min)

Po każdym nowym checkpoincie (`checkpoint-epochs=1`):
`rsync -av -e 'ssh -p <port> -i /root/.ssh/runpod_geralt_pod
-o StrictHostKeyChecking=no' root@<pod-ip>:.../checkpoints/epoch=N.ckpt
/mnt/user/geralt_voice/ckpt_backups/`, weryfikacja rozmiaru, usunięcie
poprzedniego backupu **tylko po potwierdzeniu sukcesu**. Bezpośredni rsync
Unraid↔pod (po skopiowaniu klucza SSH na Unraid) jest dużo szybszy niż
`runpodctl send/receive` (croc relay).

**Koniec sesji treningowej**: budżet $5 wyczerpany, pod RTX 5090 wyłączony
przez RunPod po **epoch=5217-step=50634.ckpt** (846MB), bezpiecznie
zbackupowany na Unraid wraz z całym datasetem (md5 zweryfikowany —
`dataset.jsonl`, `config.json`, cache, wavs identyczne z kopią roboczą).

## Etap 6 — Eksport ONNX i deploy do HA

`docker run --rm -v /mnt/user/geralt_voice:/data geralt_piper:latest
python -m piper_train.export_onnx '/data/ckpt_backups/epoch=5217-step=50634.ckpt'
/data/output/geralt.onnx`

`scripts/make_onnx_json.py` generuje `geralt.onnx.json` z `config.json`
(`num_speakers=1`, `speaker_id_map={}`, `audio.quality=medium`). Sekcja
`inference` (`length_scale: 2.3`, ustalone wcześniej dla `geralt_v2`)
skopiowana do nowego configu — Geralt mówi wolniej niż domyślny model, co
lepiej odwzorowuje oryginalny dubbing.

Deploy: `model/geralt.onnx` + `model/geralt.onnx.json` →
`/share/piper/geralt.onnx(.json)` na HA (192.168.2.200), restart addonu
`core_piper`. Stare wersje (`geralt_test`, `geralt_test2`, `geralt_v2`,
poprzedni `geralt`) usunięte — finalny model nazwany po prostu `geralt`
(widoczny w HA jako głos `"geralt"`).

Szczegóły: [`DEPLOY_HA.md`](DEPLOY_HA.md).

## Wnioski / lekcje na przyszłość

1. **RunPod container disk jest ephemeral** — nawet po "Automatic migration"
   po awarii hosta. Trzymać dataset + checkpointy na Network Volume LUB
   robić częste backupy na zewnętrzny serwer (Unraid).
2. **Network Volume ma fatalną latencję dla wielu małych plików** —
   przy starcie treningu na nowym podzie od razu kopiować dataset
   równolegle (`xargs -P 32`) na lokalny dysk kontenera, nie liczyć na
   `num_workers` samo w sobie.
3. **RTX 5090 (Blackwell, sm_120) wymaga torch ≥2.7+cu128** — domyślny obraz
   RunPod (torch 2.4.1+cu124) nie działa.
4. **pytorch-lightning 2.x wymaga manual optimization** dla VITS (GAN z
   dwoma optymizerami) i manualnego argparse/Trainer.
5. Backupy checkpointów: zawsze weryfikować sukces rsync **przed** usunięciem
   starej kopii.
