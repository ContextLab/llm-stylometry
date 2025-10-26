# Absolute File Paths - Quick Reference

All file paths use the standard bash format with complete absolute paths.

## Project Root
```
/Users/jmanning/llm-stylometry
```

## Notes Directory (Documentation)
```
/Users/jmanning/llm-stylometry/notes/
├── README.md                              # Start here - navigation guide
├── gpt2_training_pipeline_analysis.md     # Comprehensive analysis
├── training_code_snippets.md              # Code patterns
├── file_map.md                            # Codebase structure
└── absolute_file_paths.md                 # This file
```

## Main Training Pipeline Files

### Core Training Loop
```
/Users/jmanning/llm-stylometry/code/main.py
  - Lines 254-352: MAIN TRAINING LOOP (run_experiment function)
  - Line 259: model.train()
  - Lines 273-280: optimizer.zero_grad() + loss.backward() + optimizer.step()
  - Line 341: save_checkpoint()
  - Lines 348-352: Stop criteria check
```

### Model & Checkpoints
```
/Users/jmanning/llm-stylometry/code/model_utils.py
  - Lines 12-39: save_checkpoint()
  - Lines 42-98: load_checkpoint()
  - Lines 101-112: init_model()
  - Lines 115-123: count_non_embedding_params()
```

### Data Loading & Preprocessing
```
/Users/jmanning/llm-stylometry/llm_stylometry/data/loader.py
  - Lines 14-59: tokenize_texts()
  - Lines 62-145: sample_tokens()
  - Lines 148-180: OnTheFlyTrainingDataset (training)
  - Lines 182-234: EvalDataset (evaluation)
  - Lines 236-277: get_train_data_loader()
  - Lines 280-327: get_eval_data_loader()
```

### Evaluation
```
/Users/jmanning/llm-stylometry/code/eval_utils.py
  - Lines 8-37: evaluate_model()
```

### Logging
```
/Users/jmanning/llm-stylometry/code/logging_utils.py
  - update_loss_log() - appends to CSV after each epoch
```

## Configuration Files

### Experiment Configuration
```
/Users/jmanning/llm-stylometry/llm_stylometry/core/experiment.py
  - Lines 27-117: Experiment class definition
  - Properties: train_author, seed, eval_paths, stop_criteria, etc.
```

### Default Constants
```
/Users/jmanning/llm-stylometry/llm_stylometry/core/constants.py
  - Lines 44-58: DEFAULT_CONFIG
    - n_positions: 1024
    - n_embd: 128
    - batch_size: 16
    - lr: 5e-5
    - stop_criteria: {train_loss: 3.0, min_epochs: 500, max_epochs: 10000}
  - Lines 32-42: AUTHORS (8 total)
  - Lines 16-26: Directory paths
```

### Project Constants
```
/Users/jmanning/llm-stylometry/code/constants.py
  - Lines 14-24: ROOT_DIR, CODE_DIR, DATA_DIR, MODELS_DIR
  - Lines 29-39: AUTHORS list
  - Lines 42: ANALYSIS_VARIANTS
  - Lines 45-65: get_data_dir(variant)
```

### Tokenizer Utilities
```
/Users/jmanning/llm-stylometry/code/tokenizer_utils.py
  - get_tokenizer() function
```

## Data Directories

### Cleaned Training Data
```
/Users/jmanning/llm-stylometry/data/cleaned/
├── baum/              # L. Frank Baum texts
├── thompson/          # Ruth Plumly Thompson texts
├── austen/            # Jane Austen texts
├── dickens/           # Charles Dickens texts
├── fitzgerald/        # F. Scott Fitzgerald texts
├── melville/          # Herman Melville texts
├── twain/             # Mark Twain texts
├── wells/             # H.G. Wells texts
├── content_only/      # Content words variant
├── function_only/     # Function words variant
├── pos_only/          # POS tags variant
├── non_oz_baum/       # Non-Oz works by Baum
├── non_oz_thompson/   # Non-Oz works by Thompson
└── contested/         # Disputed authorship works
```

### Raw Data
```
/Users/jmanning/llm-stylometry/data/raw/
  - Original texts from Project Gutenberg
```

### Metadata
```
/Users/jmanning/llm-stylometry/data/book_and_chapter_titles.txt
  - Used for cleaning text during preprocessing
```

## Model Storage

### Trained Models Directory
```
/Users/jmanning/llm-stylometry/models/
├── baum_tokenizer=gpt2_seed=0/
├── baum_tokenizer=gpt2_seed=1/
├── ...
├── wells_tokenizer=gpt2_seed=9/
└── {author}_variant={variant}_tokenizer=gpt2_seed={seed}/
    (one per author, seed, variant combination)
```

### Model Files (Inside Each Model Directory)
```
/Users/jmanning/llm-stylometry/models/{model_name}/
├── config.json                 # Model architecture config
├── generation_config.json      # Generation settings
├── model.safetensors          # Model weights (HuggingFace format)
├── training_state.pt          # CRITICAL: Optimizer + RNG states
└── loss_logs.csv              # Training history log
```

## Test & Development Files

### Tests
```
/Users/jmanning/llm-stylometry/tests/
├── test_model_training.py      # Training tests
├── test_variant_training.py     # Variant training tests
├── test_variant_data_loading.py # Data loading tests
└── conftest.py                  # Test configuration
```

## Shell Scripts

### Main Training Script
```
/Users/jmanning/llm-stylometry/run_llm_stylometry.sh
  - Entry point for training
  - Handles environment variables
```

### Remote Training Script
```
/Users/jmanning/llm-stylometry/remote_train.sh
  - SSH-based remote training
  - Backup/restore functionality
```

### Sync Models
```
/Users/jmanning/llm-stylometry/sync_models.sh
  - Download trained models from remote server
```

## Documentation Files

### Project Documentation
```
/Users/jmanning/llm-stylometry/CLAUDE.md
  - Project README with key commands
  - Architecture overview
  - Troubleshooting guide
```

### Global Development Guidelines
```
/Users/jmanning/.claude/CLAUDE.md
  - Global best practices
  - Testing methodology
  - Commit guidelines
```

## Important CSV Output Files

### Loss Logs
```
/Users/jmanning/llm-stylometry/models/{model_name}/loss_logs.csv
  Columns: epochs_completed, loss_dataset, loss_value, seed, train_author
  - One row per evaluation dataset per epoch
  - Used to track convergence and check stop criteria
```

### Model Results (Aggregated)
```
/Users/jmanning/llm-stylometry/data/model_results.pkl
  - Pandas DataFrame with all loss logs
  - Used for visualization and analysis
```

## Quick Command Reference

### View Notes
```bash
cat /Users/jmanning/llm-stylometry/notes/README.md
cat /Users/jmanning/llm-stylometry/notes/gpt2_training_pipeline_analysis.md
cat /Users/jmanning/llm-stylometry/notes/training_code_snippets.md
cat /Users/jmanning/llm-stylometry/notes/file_map.md
```

### Check Training Progress
```bash
tail -f /Users/jmanning/llm-stylometry/models/*/loss_logs.csv
```

### Run Training
```bash
cd /Users/jmanning/llm-stylometry
python code/main.py
```

### Resume Training
```bash
cd /Users/jmanning/llm-stylometry
RESUME_TRAINING=1 python code/main.py
```

### Check Model Status
```bash
python /Users/jmanning/llm-stylometry/code/check_training_status.py
```

## File Dependencies Graph

```
code/main.py
  ├─ imports from code/:
  │  ├─ model_utils.py
  │  ├─ data_utils.py
  │  ├─ eval_utils.py
  │  ├─ logging_utils.py
  │  ├─ tokenizer_utils.py
  │  ├─ experiment.py
  │  └─ constants.py
  │
  └─ imports from transformers:
     ├─ GPT2Config
     └─ GPT2LMHeadModel
```

## Python Import Paths

All Python files use relative imports from the project root.

### From code/main.py:
```python
from model_utils import save_checkpoint, load_checkpoint, init_model
from data_utils import get_train_data_loader, get_eval_data_loader
from eval_utils import evaluate_model
from experiment import Experiment
from constants import MODELS_DIR, AUTHORS
```

### From code/:
```python
from constants import CLEANED_DATA_DIR, AUTHORS, MODELS_DIR
from tokenizer_utils import get_tokenizer
```

### From llm_stylometry/data/loader.py:
```python
from ..core.constants import CLEANED_DATA_DIR
```

### From llm_stylometry/core/experiment.py:
```python
from ..core.constants import AUTHORS, CLEANED_DATA_DIR, DEFAULT_CONFIG
```

## Environment Variables

### Training Control
```bash
RESUME_TRAINING=1          # Resume from checkpoint
ANALYSIS_VARIANT=content   # or 'function', 'pos', or unset for baseline
NO_MULTIPROCESSING=1       # Run sequentially instead of parallel
MAX_GPUS=2                 # Limit number of GPUs to use
DISABLE_TQDM=1             # Disable progress bars
```

---

Last updated: 2025-10-24
