import json
c = json.load(open('/data/training_processed/config.json'))
c['num_speakers'] = 1
c['speaker_id_map'] = {}
c['audio']['quality'] = 'medium'
json.dump(c, open('/data/output/geralt.onnx.json', 'w'), ensure_ascii=False, indent=2)
print('written')
