# Models Directory

Contains 320 trained GPT-2 models (80 per condition: baseline, content-only, function-only, POS).

## Directory Naming

**Baseline:** `{author}_tokenizer=gpt2_seed={0-9}/`
**Variants:** `{author}_variant={variant}_tokenizer=gpt2_seed={0-9}/`

Examples:
- `baum_tokenizer=gpt2_seed=0/` (baseline)
- `austen_variant=content_tokenizer=gpt2_seed=5/` (content-only)

## File Contents

Each directory contains:
- `config.json`, `generation_config.json` - Model configuration
- `loss_logs.csv` - Training/evaluation losses per epoch
- `model.safetensors` - Model weights (~32MB, gitignored)
- `training_state.pt` - Optimizer state (~65MB, gitignored)

**Note:** Weight files are gitignored due to size. Download pre-trained weights to use or explore trained models (not required for generating figures).

## Downloading Pre-trained Weights

Model weight files (`.safetensors`, `training_state.pt`) are gitignored due to size (~30GB total). Download pre-trained weights from Dropbox:

```bash
# Download all variants (~26.6GB compressed, ~30GB extracted)
./download_model_weights.sh --all

# Download specific variants
./download_model_weights.sh -b           # Baseline only (~6.7GB)
./download_model_weights.sh -co -fo      # Content + function (~13.4GB)
```

**Archive details:**
- `model_weights_baseline.tar.gz` - 80 baseline models (6.7GB compressed)
- `model_weights_content.tar.gz` - 80 content-only models (6.7GB compressed)
- `model_weights_function.tar.gz` - 80 function-only models (6.6GB compressed)
- `model_weights_pos.tar.gz` - 80 POS models (6.6GB compressed)

Each archive is verified with SHA256 checksums (checked into git). The download script automatically:
- Downloads from Dropbox with resume support
- Verifies file integrity via SHA256
- Extracts to correct model directories
- Validates all 80 models per variant

**Note:** Pre-trained weights are only needed to explore trained models or run inference. All paper figures can be generated from `data/model_results*.pkl` files without downloading weights.

## Training Models

Train locally:
```bash
./run_llm_stylometry.sh --train           # Baseline
./run_llm_stylometry.sh --train -co       # Content-only
```

Train remotely on GPU cluster:
```bash
./remote_train.sh                         # Baseline
./remote_train.sh -co --cluster tensor02  # Content-only on tensor02
```

See main README for full training documentation.

## HuggingFace Model Training (High-Quality Public Models)

Train high-quality single models (one per author) for public release on HuggingFace.

### Training Commands

**Local training:**
```bash
./train_hf_models.sh --author baum           # Single author, 50k epochs
./train_hf_models.sh --all                   # All 8 authors in parallel
```

**Remote GPU training:**
```bash
./remote_train_hf.sh --cluster mycluster --all              # All 8 authors
./remote_train_hf.sh --cluster mycluster --author baum      # Single author
./remote_train_hf.sh --cluster mycluster --all --max-epochs 100000  # Higher limit
```

**Monitor training:**
```bash
./check_hf_status.sh --cluster mycluster
# Shows current epoch, loss, progress, ETA per author
```

**Download completed models:**
```bash
./sync_hf_models.sh --cluster mycluster --all
# Downloads to models_hf/{author}_tokenizer=gpt2/
```

### Uploading to HuggingFace

**Prerequisites:**
- Credentials file: `.huggingface/credentials.json`
- Format: `{"username": "contextlab", "token": "hf_..."}`
- Completed models in `models_hf/` directory

**Generate model cards:**
```bash
python code/generate_model_card.py --author baum --model-dir models_hf/baum_tokenizer=gpt2
```

**Upload models:**
```bash
./upload_to_huggingface.sh --author baum --dry-run    # Test
./upload_to_huggingface.sh --author baum              # Upload
./upload_to_huggingface.sh --all                      # Upload all
```

Models published to: `contextlab/gpt2-{author}` (e.g., contextlab/gpt2-baum)

### HuggingFace vs Paper Models

**Paper models:** 320 models (8 authors × 10 seeds × 4 conditions)
- Used for all figures and statistical analysis
- Trained to loss ≤ 3.0 for consistent comparison
- Available via Dropbox download

**HuggingFace models:** 8 models (1 per author)
- For public use and text generation
- Trained for 50,000 additional epochs beyond paper models
- Much lower loss (~1.3-1.6) for better generation quality

Trained models available on HuggingFace:
- Jane Austen: [contextlab/gpt2-austen](https://huggingface.co/contextlab/gpt2-austen)
- L. Frank Baum: [contextlab/gpt2-baum](https://huggingface.co/contextlab/gpt2-baum) (training)
- Charles Dickens: [contextlab/gpt2-dickens](https://huggingface.co/contextlab/gpt2-dickens)
- F. Scott Fitzgerald: [contextlab/gpt2-fitzgerald](https://huggingface.co/contextlab/gpt2-fitzgerald)
- Herman Melville: [contextlab/gpt2-melville](https://huggingface.co/contextlab/gpt2-melville)
- Ruth Plumly Thompson: [contextlab/gpt2-thompson](https://huggingface.co/contextlab/gpt2-thompson)
- Mark Twain: [contextlab/gpt2-twain](https://huggingface.co/contextlab/gpt2-twain)
- H.G. Wells: [contextlab/gpt2-wells](https://huggingface.co/contextlab/gpt2-wells)
