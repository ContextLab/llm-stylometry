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

**Note:** Weight files are gitignored due to size. See issue #36 for downloading pre-trained weights.

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
