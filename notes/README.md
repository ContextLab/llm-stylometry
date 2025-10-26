# GPT-2 Training Pipeline Analysis - Notes

This directory contains comprehensive documentation of the GPT-2 model training pipeline in the llm-stylometry project.

## Files in This Directory

### 1. **gpt2_training_pipeline_analysis.md** (Comprehensive Overview)
The main reference document covering:
- Key files and their roles
- The complete training loop (lines 254-352 of main.py)
- Stop criteria definition and enforcement
- Checkpoint saving and loading
- Experiment configuration
- Data loading pipeline (training vs evaluation)
- Evaluation and logging
- Resume training logic
- Device handling
- Analysis variants
- Architecture summary
- Extension points for HuggingFace

**Start here** if you need a complete understanding of the system.

### 2. **training_code_snippets.md** (Quick Code Reference)
Extracted code snippets with line numbers for:
1. Complete training loop (simplified)
2. Checkpoint saving
3. Checkpoint loading (resume)
4. Model initialization
5. Stop criteria check
6. Training data loader
7. Evaluation data loader
8. Model evaluation function
9. Resume training logic
10. Multiprocessing setup
11. Mixed precision training
12. Device detection
13. Experiment creation
14. Key variables and constants
15. Environment variables

**Use this** when you need to see specific code patterns.

### 3. **file_map.md** (Codebase Structure)
Visual maps of:
- Directory structure (training-relevant files)
- Call graph of the training pipeline
- File dependencies
- Critical files by function
- Key variable flow
- Resume training flow diagram

**Use this** to understand file organization and dependencies.

---

## Quick Start: Finding What You Need

### If you want to understand...

**The main training loop:**
- See: `gpt2_training_pipeline_analysis.md` → "THE MAIN TRAINING LOOP" section
- Or: `training_code_snippets.md` → Code snippet #1
- Or: `file_map.md` → "Call Graph: Training Pipeline"

**How checkpoints work:**
- See: `gpt2_training_pipeline_analysis.md` → "Checkpoint Saving & Loading" section
- Or: `training_code_snippets.md` → Code snippets #2 and #3

**When training stops:**
- See: `gpt2_training_pipeline_analysis.md` → "Stop Criteria & Early Stopping" section
- Or: `training_code_snippets.md` → Code snippet #5

**How to resume training:**
- See: `gpt2_training_pipeline_analysis.md` → "Resume Training Logic" section
- Or: `training_code_snippets.md` → Code snippet #9
- Or: `file_map.md` → "Resume Training Flow"

**The file structure:**
- See: `file_map.md` → "Directory Structure" and "File Dependencies"

**Critical files to modify:**
- See: `file_map.md` → "Critical Files by Function"

---

## Key Findings Summary

### Main Training Loop Location
**File:** `/code/main.py`, Lines 254-352  
**Function:** `run_experiment(exp, device_queue, device_type)`

Key operations:
- `model.train()` - Sets training mode
- `optimizer.zero_grad()` - Clears gradients
- `loss.backward()` - Computes gradients
- `optimizer.step()` - Updates weights
- `save_checkpoint()` - Saves after each epoch
- Stop check - Breaks if loss ≤ 3.0 AND epochs ≥ 500

### Stop Criteria
```python
stop_criteria = {
    "train_loss": 3.0,      # Must reach this loss level
    "min_epochs": 500,      # Must train at least this many epochs
    "max_epochs": 10000,    # Hard limit
}
```

**Enforcement:** Lines 348-352 of `/code/main.py`
```python
if train_loss <= stop_train_loss and min_epochs <= epochs_completed:
    break  # Exit training loop
```

### Checkpointing
- **Saves after each epoch** (Line 341 of `/code/main.py`)
- **What gets saved:**
  1. Model weights (HF format): `model.safetensors`, `config.json`
  2. Training state: `training_state.pt` containing:
     - Optimizer state (for momentum)
     - Current epoch number
     - All RNG states (for reproducible resume)

### Resume Training
Three scenarios:
1. **Training complete** (loss ≤ 3.0 AND epochs ≥ 500) → Skip
2. **Has weights + training_state.pt** → Resume from checkpoint
3. **No weights** → Start fresh (removes old logs)

### Multiprocessing
- Uses `mp.Pool` with one process per GPU
- Device queue manages GPU assignment
- Falls back to sequential if needed

### Data Loading
- **Training:** On-the-fly random sampling (shuffle=True)
  - Sample 643,041 tokens per author
  - Random subsequences each batch for diversity
- **Evaluation:** Sequential non-overlapping chunks (shuffle=False)
  - Ensure full book coverage without repeats

---

## For Extending to HuggingFace Training

Key modules to understand/modify:

1. **Model Initialization** (`/code/model_utils.py`)
   - Currently: `GPT2LMHeadModel(config).to(device)`
   - For HF: `AutoModelForCausalLM.from_pretrained()`

2. **Training Loop** (`/code/main.py`, lines 254-352)
   - Currently: Manual loop with mixed precision
   - Could use: HF `Trainer` or keep manual loop

3. **Checkpoint Format** (`/code/model_utils.py`)
   - Currently: Uses `.safetensors` via `save_pretrained()`
   - Already HF compatible!

4. **Stop Criteria** (`/llm_stylometry/core/constants.py`)
   - Currently: Manual check each epoch
   - For HF: Use `callbacks` or `save_strategy`

---

## File Locations (Absolute Paths)

All files are in: `/Users/jmanning/llm-stylometry/`

### Training Pipeline Files
- `/Users/jmanning/llm-stylometry/code/main.py` - Main training loop
- `/Users/jmanning/llm-stylometry/code/model_utils.py` - Checkpoints
- `/Users/jmanning/llm-stylometry/llm_stylometry/data/loader.py` - Data loading
- `/Users/jmanning/llm-stylometry/code/eval_utils.py` - Evaluation
- `/Users/jmanning/llm-stylometry/llm_stylometry/core/experiment.py` - Config

### Constants Files
- `/Users/jmanning/llm-stylometry/llm_stylometry/core/constants.py` - DEFAULT_CONFIG
- `/Users/jmanning/llm-stylometry/code/constants.py` - Project constants

---

## Commands for Testing

Check training progress:
```bash
tail -f /Users/jmanning/llm-stylometry/models/*/loss_logs.csv
```

Resume training:
```bash
cd /Users/jmanning/llm-stylometry
RESUME_TRAINING=1 python code/main.py
```

Check specific model:
```bash
python code/check_training_status.py baum_tokenizer=gpt2_seed=0
```

---

## Document Generation

These documents were generated by analyzing the codebase on 2025-10-24 using:
- Direct file reading of key training files
- Grep searches for specific patterns
- Analysis of file dependencies and call graphs

All information is accurate as of the last git status update.

---

## Related Documentation

See also:
- `/Users/jmanning/llm-stylometry/CLAUDE.md` - Project README
- `/Users/jmanning/.claude/CLAUDE.md` - Global development guidelines

---

Last updated: 2025-10-24
