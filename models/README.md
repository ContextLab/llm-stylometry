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
