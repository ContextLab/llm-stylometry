# GPT-2 Model Training Pipeline Analysis

## Overview
This document maps the complete GPT-2 training pipeline in the llm-stylometry project, showing where the main training loop is, how checkpoints are saved, and how stop criteria are enforced.

**Date:** 2025-10-24  
**Codebase:** /Users/jmanning/llm-stylometry

---

## Key Files & Their Roles

### 1. **Entry Point: `/code/main.py`** (Lines 363-468)
The main training orchestrator.

**Key Functions:**
- `run_experiment(exp: Experiment, device_queue, device_type="cuda")` (Lines 132-360)
  - **This is the MAIN TRAINING FUNCTION**
  - Called once per experiment (80 models: 8 authors × 10 seeds)
  - Runs in separate process when multiprocessing enabled

**Multiprocessing Logic:**
- Detects CUDA device count (Lines 86-98)
- Creates device queue for GPU assignment
- Uses `mp.Pool` with processes=gpu_count (Line 434)
- Falls back to sequential mode if needed (Lines 449-467)

---

## THE MAIN TRAINING LOOP

Located in: `/code/main.py`, Lines 254-352 in `run_experiment()`

```python
# Training loop (Line 254)
for epoch in tqdm(range(start_epoch, max_epochs)):
    total_train_loss = 0.0
    
    # Iterate over batches in the training dataloader (Line 258)
    for batch_idx, batch in enumerate(train_dataloader):
        model.train()  # Line 259
        
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass (Lines 264-270)
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
        
        # Backward pass with optimizer.step() (Lines 272-280)
        optimizer.zero_grad()  # Line 273
        if use_amp:
            scaler.scale(loss).backward()  # Line 275
            scaler.step(optimizer)         # Line 276 - CRITICAL: Updates weights
            scaler.update()                 # Line 277
        else:
            loss.backward()                 # Line 279
            optimizer.step()                # Line 280 - CRITICAL: Updates weights
```

**Key Points:**
- `model.train()` sets training mode (Line 259)
- `optimizer.step()` updates model weights (Line 280 or 276)
- Mixed precision training with gradient scaling on CUDA (Lines 264-277)
- Loss accumulation for epoch average (Line 283)

---

## Stop Criteria & Early Stopping

### Definition
Located in: `/llm_stylometry/core/constants.py`, Lines 53-57

```python
"stop_criteria": {
    "train_loss": 3.0,        # Target training loss threshold
    "min_epochs": 500,        # Minimum epochs before stopping allowed
    "max_epochs": 10000,      # Hard limit on training iterations
}
```

### Enforcement
Located in: `/code/main.py`, Lines 196-201 and Lines 348-352

```python
# Retrieve stop criteria (Lines 196-201)
stop_train_loss = exp.stop_criteria["train_loss"]   # 3.0
min_epochs = exp.stop_criteria["min_epochs"]         # 500
max_epochs = exp.stop_criteria["max_epochs"]         # 10000

# Early stopping check (Lines 348-352) - runs AFTER each epoch
if train_loss <= stop_train_loss and min_epochs <= epochs_completed:
    logger.info(
        f"Training loss {train_loss:.4f} below threshold {stop_train_loss}. "
        f"Stopping training."
    )
    break  # Exit training loop
```

**Logic:**
1. After each epoch, compute average training loss (Line 295)
2. Check: Is loss ≤ 3.0 AND epochs ≥ 500?
3. If YES: Stop training and exit loop
4. If NO: Continue until max_epochs (10,000)

---

## Checkpoint Saving & Loading

### Saving Checkpoints
Located in: `/code/model_utils.py`, Lines 12-39 in `save_checkpoint()`

**Called after each epoch** (Line 341 in main.py):
```python
save_checkpoint(
    model=model,
    optimizer=optimizer,
    model_name=modelname,
    epochs_completed=epochs_completed,
)
```

**What Gets Saved:**

1. **Model weights** (Line 21)
   ```python
   model.save_pretrained(save_directory=checkpoint_dir)
   # Creates: config.json, generation_config.json, model.safetensors
   ```

2. **Training state** (Lines 24-36) - Saved to `training_state.pt`
   ```python
   training_state = {
       "optimizer_state_dict": optimizer.state_dict(),      # AdamW state
       "epochs_completed": epochs_completed,                # Current epoch number
       "random_state": random.getstate(),                   # Python RNG
       "np_random_state": np.random.get_state(),            # NumPy RNG
       "torch_random_state": torch.get_rng_state(),         # PyTorch RNG
       "cuda_random_state": torch.cuda.get_rng_state_all(), # CUDA RNG
   }
   torch.save(training_state, checkpoint_dir / "training_state.pt")
   ```

**Directory Structure:**
```
models/{model_name}/
├── config.json                 # Model architecture
├── generation_config.json      # Generation settings
├── model.safetensors          # Weights (HF format)
├── training_state.pt          # Optimizer + RNG states
└── loss_logs.csv              # Training history
```

### Loading Checkpoints (Resume Training)
Located in: `/code/model_utils.py`, Lines 42-98 in `load_checkpoint()`

**Called if `resume_training=True`** (Lines 204-209 in main.py):
```python
model, optimizer, start_epoch = load_checkpoint(
    model_class=GPT2LMHeadModel,
    model_name=modelname,
    device=device,
)
```

**What Gets Loaded:**
1. Model weights from HF format (Lines 48-50)
2. AdamW optimizer state (Lines 58-59)
3. Epoch number to resume from (Line 60)
4. All RNG states for deterministic continuation (Lines 63-92)

**Critical: Training state contains optimizer state!**
- Without `training_state.pt`, resume loses momentum and learning rate schedule
- All random seeds restored for reproducibility

---

## Experiment Configuration

Located in: `/llm_stylometry/core/experiment.py`

**Experiment Class** holds:
- `train_author`: Which author's texts to train on
- `seed`: Random seed for reproducibility
- `eval_paths`: Dictionary of evaluation datasets for each author
- Training hyperparameters (n_positions, n_embd, n_layer, etc.)
- `stop_criteria`: Dict with train_loss, min_epochs, max_epochs
- `resume_training`: Boolean flag

**Experiment Creation** (Lines 119-129 in main.py):
```python
experiments = []
for seed in range(10):
    for author in AUTHORS:
        experiments.append(
            Experiment(
                train_author=author,
                seed=seed,
                tokenizer_name="gpt2",
                analysis_variant=variant,  # Optional: 'content', 'function', 'pos'
                resume_training=resume_mode,
            )
        )
# Creates 80 experiments (8 authors × 10 seeds)
```

---

## Data Loading Pipeline

Located in: `/llm_stylometry/data/loader.py`

### Training Data: On-the-Fly Sampling
Called in main.py (Lines 163-171):

```python
train_dataloader = get_train_data_loader(
    path=exp.data_dir / exp.train_author,
    tokenizer=tokenizer,
    n_positions=1024,              # Context length
    batch_size=16,                 # Batch size
    n_tokens=643041,               # Total tokens per author
    seed=exp.seed,                 # Reproducible sampling
    excluded_train_path=exp.excluded_train_path,
)
```

**How it works:**
1. `tokenize_texts()` - Loads all author's books, tokenizes them
2. `sample_tokens()` - Samples 643,041 tokens proportionally from all books
3. `OnTheFlyTrainingDataset` - Creates dataset with n_samples = n_tokens / n_positions
   - Returns random subsequence for each batch (not sequential)
   - Ensures diversity within epoch

### Evaluation Data: Sequential Processing
Called in main.py (Lines 177-185):

```python
eval_dataloaders = {
    name: get_eval_data_loader(
        path=path,
        tokenizer=tokenizer,
        n_positions=1024,
        batch_size=16,
    )
    for name, path in exp.eval_paths.items()
}
# One dataloader per evaluation author + special sets (non_oz, contested)
```

**How it works:**
1. `EvalDataset` - Loads single book, chunks into non-overlapping sequences
2. Sequential chunks ensure full book coverage with no repeats/gaps

---

## Evaluation & Logging

Located in: `/code/eval_utils.py`, Lines 8-37

**evaluate_model()** called after each epoch for each eval dataset:
```python
def evaluate_model(model, eval_dataloader, device):
    model.eval()                    # Set eval mode
    total_loss = 0.0
    
    with torch.no_grad():           # Disable gradients
        for batch in eval_dataloader:
            with torch.amp.autocast(...):  # Mixed precision
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss.item()
                total_loss += loss
    
    return total_loss / len(eval_dataloader)  # Average loss
```

**Loss Logging** (`logging_utils.py`):
- After each epoch, calls `update_loss_log()` for:
  - Training loss (1 entry per epoch)
  - Evaluation loss for each author (up to 11 datasets per epoch)
- CSV format: epoch, loss_dataset, loss_value, seed, author

---

## Resume Training Logic

Located in: `/code/main.py`, Lines 368-408

**Decision Tree:**
1. **Is complete?** (training_loss ≤ 3.0 AND epochs ≥ 500)
   - Skip: Already done training
   
2. **Has weights + training_state.pt?**
   - Resume from checkpoint
   
3. **Has loss_logs.csv but NO weights?** (e.g., after git clone)
   - Delete model dir, start fresh
   
4. **No logs or weights?**
   - Start fresh

```python
is_complete, has_weights, epochs_done = check_model_complete(
    exp.name,
    exp.stop_criteria["train_loss"],
    exp.stop_criteria["min_epochs"]
)
```

---

## Device Handling

**Automatic Detection** (Lines 86-98):
- CUDA: Use all GPUs (or MAX_GPUS if limited)
- MPS: Apple Metal Performance Shaders (1 device)
- CPU: Fallback (slow)

**Mixed Precision Training** (Lines 243-277):
- Only on CUDA: Automatic gradient scaling with float16
- Forward pass: `torch.amp.autocast(device_type='cuda', dtype=torch.float16)`
- Backward: `scaler.scale(loss).backward()`

**Memory Management** (Lines 288-290):
- Delete intermediate tensors after backward pass
- Clear CUDA cache every 5 batches
- Gradient checkpointing enabled (Line 248)

---

## Analysis Variants

**Supported Variants:**
- `None` (baseline): All words
- `'content'`: Content words only (function words masked)
- `'function'`: Function words only
- `'pos'`: POS tags only

**Data Directory Routing** (`constants.py`):
```python
def get_data_dir(variant=None):
    if variant is None:
        return data/cleaned/              # baseline
    else:
        return data/cleaned/{variant}_only/  # variant
```

**Model Naming:**
- Baseline: `{author}_tokenizer=gpt2_seed={seed}`
- Variant: `{author}_variant={variant}_tokenizer=gpt2_seed={seed}`

---

## Architecture Summary: How Everything Connects

```
main.py
  └─ run_experiment() [Main training loop]
     ├─ Load/init model (model_utils.py)
     │  ├─ init_model() - Create new model with GPT2Config
     │  └─ load_checkpoint() - Load saved weights + optimizer state
     │
     ├─ Data loading (data/loader.py)
     │  ├─ get_train_data_loader() - On-the-fly sampling
     │  └─ get_eval_data_loader() - Sequential evaluation
     │
     ├─ Training loop (main.py:254-352)
     │  ├─ model.train()
     │  ├─ Forward: model(input_ids)
     │  ├─ Backward: optimizer.step()
     │  ├─ Evaluate each dataset (eval_utils.py)
     │  ├─ Log losses (logging_utils.py)
     │  ├─ Save checkpoint (model_utils.py)
     │  └─ Check stop criteria
     │
     └─ Resume capability
        └─ load_checkpoint() restores everything needed

Experiment class (core/experiment.py)
  └─ Defines: author, seed, hyperparams, stop_criteria, eval_paths
```

---

## Key Stopping Behaviors

| Condition | Behavior |
|-----------|----------|
| `train_loss ≤ 3.0 AND epochs ≥ 500` | **Stop training, keep checkpoint** |
| `epochs == 10,000` | **Hard stop (reached max)** |
| `resume_training=True` AND weights exist | **Continue from saved epoch** |
| `resume_training=True` AND NO weights | **Start fresh** |

---

## Extension Points for HuggingFace Training

Based on this analysis, here are key areas to modify for HuggingFace model training:

1. **Model Initialization** (`model_utils.py`):
   - Currently: `GPT2LMHeadModel(config).to(device)`
   - For HF: Use `AutoModelForCausalLM.from_pretrained()`

2. **Optimizer Setup** (`model_utils.py`):
   - Currently: `AdamW` with fixed learning rate
   - For HF: Could use `Trainer` with learning rate schedule

3. **Training Loop** (`main.py`):
   - Currently: Manual loop with mixed precision
   - For HF: Could use `Trainer.train()` or keep manual loop

4. **Checkpoint Format** (`model_utils.py`):
   - Currently: Uses `.safetensors` via `save_pretrained()`
   - For HF: Already compatible! Just ensure `training_state.pt` is saved

5. **Stop Criteria Integration**:
   - Currently: Checked after each epoch manually
   - For HF Trainer: Would use `callbacks` or `save_strategy`

---

## Files to Study Next

For extending to HuggingFace training:
1. `/code/main.py` - Understand the training loop structure
2. `/code/model_utils.py` - Understand checkpoint save/load
3. `/llm_stylometry/core/experiment.py` - Configuration object
4. `/llm_stylometry/data/loader.py` - Data pipeline

---

## Useful Commands

Check training progress:
```bash
tail -f models/*/loss_logs.csv
```

Resume training:
```bash
RESUME_TRAINING=1 python code/main.py
```

Check specific model status:
```bash
python code/check_training_status.py baum_tokenizer=gpt2_seed=0
```

---

Generated: 2025-10-24
