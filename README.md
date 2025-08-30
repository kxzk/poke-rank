<div align="center"><h3>PokéRank</h3></div>

> Fine-tune a compact LLM to tier-rank Pokémon Pocket TCG cards (S, A+, A, B, C, D)

## Overview

This project trains a Qwen 2.5 1.5B model using LoRA to classify Pokémon Pocket cards into competitive tiers. The model learns from annotated card data, analyzing stats, abilities, and game mechanics to predict viability.

## Setup

```bash
# fetch card data from api
make get-raw-data

# prepare for annotation
make prep-data
```

## Data Pipeline

### 1. Acquisition
Two methods to obtain card data:

**API method** (recommended):
```bash
make get-raw-data
```

**Browser cache method**:
- Visit [ptcgpocket.gg](https://ptcgpocket.gg/cards/)
- Open DevTools → Network tab → locate cache number
- Run: `uv run scripts/get_card_data.py -c <cache_number>`

### 2. Processing
```bash
# compress card data for model input
uv run scripts/compress_cards.py

# format for annotation
python3 scripts/prepare_for_annotation.py
```

### 3. Annotation
Label cards in `data/annotated_pokemon.json` with tiers:
- **S**: meta-defining, essential
- **A+**: highly competitive, versatile
- **A**: solid competitive choice
- **B**: situational, niche use
- **C**: casual play viable
- **D**: collection only

## Training

Configuration in `train.py`:
- Model: Qwen 2.5 1.5B (4-bit quantized)
- Method: LoRA fine-tuning (r=16)
- Max sequence: 1024 tokens
- Output: `outputs/` directory

## Architecture

```
├── data/
│   ├── annotated_pokemon.json   # labeled training data
│   ├── cards_compressed.txt     # processed card features
│   └── cards_*.csv              # raw card database
├── scripts/
│   ├── get_card_data.py         # scraper for card stats
│   ├── compress_cards.py        # feature extraction
│   └── prepare_for_annotation.py # format for labeling
└── train.py                      # fine-tuning pipeline
```

<br>
