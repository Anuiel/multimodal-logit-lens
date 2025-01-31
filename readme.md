# Project Title

This projects dives deep into Qwen2.5-VL-2B and using logit lens method found some interesting patters using [TallyQA](https://www.manojacharya.com/tallyqa.html) dataset.

## Installation

Project based on Python 3.11.5. Make sure to install requirements 
```bash
python3 install -r requirements
```

To reproduse result you need to download TallyQA and The Visual Genome. Because I only used test part this process is simplier then in TallyQA repo. 
```sh
wget -O data/tallyqa.zip "https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true"
unzip data/tallyqa.zip

mkdir data/VG_100K_2
wget -O data/VG_100K_2/images.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget -O data/VG_100K_2/images2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip -q data/VG_100K_2/images.zip && unzip -q data/VG_100K_2/images2.zip
mv data/VG_100K_2/VG_100K_2/* data/VG_100K_2/ && mv data/VG_100K_2/VG_100K/* data/VG_100K_2/
rm -rf data/VG_100K_2/VG_100K_2 data/VG_100K_2/VG_100K
rm -f data/VG_100K_2/images2.zip data/VG_100K_2/images.zip
```

## Reproduse

To reproduse result first you need to exctract logits' features using [src/get_logits.py](src/get_logits.py) script. Simply run
```sh
python3 src/get_logits.py
```

Be aware that script uses `cuda:0` device and require at least 12 Gb VRAM GPU.

To see pictures run
```sh
python3 src/parse_logits.py --logits-path logit.json --output-path figures --chinise-mapping-path number_to_chinise_filtered.json
python3 src/example_showcase.py --logits-path logit.json --output-path figures
```

## Result

Read more in [result.md](result.md)