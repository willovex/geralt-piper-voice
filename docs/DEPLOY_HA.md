# Wgranie / aktualizacja głosu w Home Assistant

Home Assistant ma zainstalowany addon **Piper** (Wyoming protocol, slug
`core_piper`). Każda para plików `<nazwa>.onnx` + `<nazwa>.onnx.json` wgrana
do `/share/piper/` staje się dostępnym głosem `<nazwa>` (bez rozszerzenia).

## 1. Eksport checkpointu do ONNX

Na serwerze z dostępem do checkpointu (Unraid, `/mnt/user/geralt_voice/`):

```bash
docker run --rm -v /mnt/user/geralt_voice:/data geralt_piper:latest \
  python -m piper_train.export_onnx \
  '/data/ckpt_backups/epoch=XXXX-step=YYYYYY.ckpt' \
  /data/output/geralt.onnx
```

(Obraz `geralt_piper:latest` zbudowany z `scripts/Dockerfile.piper`.)

## 2. Wygenerowanie `.onnx.json`

```bash
docker run --rm -v /mnt/user/geralt_voice:/data geralt_piper:latest \
  python /data/scripts/make_onnx_json.py
```

`scripts/make_onnx_json.py` bierze `training_processed/config.json`, ustawia
`num_speakers=1`, `speaker_id_map={}`, `audio.quality="medium"` i zapisuje
`output/geralt.onnx.json`.

**Ważne**: dopisz/zachowaj sekcję `inference` — bez niej Piper użyje
domyślnego `length_scale=1.0` i Geralt będzie mówił zbyt szybko:

```json
"inference": {
  "noise_scale": 0.667,
  "length_scale": 2.3,
  "noise_w": 0.8
}
```

## 3. Wgranie do HA

Skopiuj oba pliki do `/share/piper/` na HA (192.168.2.200), np.:

```bash
scp output/geralt.onnx output/geralt.onnx.json root@192.168.2.200:/share/piper/
```

## 4. Restart addonu Piper

```bash
ssh root@192.168.2.200 ha addons restart core_piper
```

Po restarcie nowy/zaktualizowany głos `geralt` jest widoczny na liście
głosów Piper.

## 5. Użycie w automatyzacjach / skryptach

```yaml
action: tts.speak
target:
  entity_id: tts.piper
data:
  media_player_entity_id: media_player.salon
  message: "Witaj, wędrowcze."
  options:
    voice: geralt
```

**Uwaga**: `voice` musi być pod `data.options.voice`, NIE zagnieżdżone w
`message`.

## Aktualna wersja

- Model: `model/geralt.onnx` (checkpoint `epoch=5217-step=50634`)
- Wdrożony jako `/share/piper/geralt.onnx` + `geralt.onnx.json` na HA
- Stare warianty (`geralt_test`, `geralt_test2`, `geralt_v2`) usunięte —
  jeden głos `geralt`
