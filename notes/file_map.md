# File Map: GPT-2 Training Pipeline

## Directory Structure (Training-Relevant Files)

```
llm-stylometry/
│
├── code/
│   ├── main.py                      [MAIN TRAINING LOOP]
│   │   ├── run_experiment()         Main function (per-model training)
│   │   ├── check_model_complete()   Resume logic
│   │   └── get_device_info()        Device detection
│   │
│   ├── model_utils.py
│   │   ├── save_checkpoint()        Saves weights + training state
│   │   ├── load_checkpoint()        Loads for resume
│   │   ├── init_model()             Creates new model
│   │   └── count_non_embedding_params()
│   │
│   ├── data_utils.py
│   │   ├── tokenize_texts()         Load texts, tokenize
│   │   ├── sample_tokens()          Sample proportionally
│   │   ├── get_train_data_loader()  Create training dataloader
│   │   └── get_eval_data_loader()   Create eval dataloader
│   │
│   ├── eval_utils.py
│   │   └── evaluate_model()         Compute loss per epoch
│   │
│   ├── experiment.py
│   │   └── Experiment class         Configuration object
│   │
│   ├── constants.py
│   │   ├── AUTHORS
│   │   ├── MODELS_DIR
│   │   ├── CLEANED_DATA_DIR
│   │   └── get_data_dir()
│   │
│   ├── tokenizer_utils.py
│   ├── logging_utils.py
│   └── main.py
│
├── llm_stylometry/
│   ├── core/
│   │   ├── constants.py             [DEFAULT_CONFIG, STOP_CRITERIA]
│   │   │   ├── DEFAULT_CONFIG       Default hyperparameters
│   │   │   │   ├── n_positions: 1024
│   │   │   │   ├── n_embd: 128
│   │   │   │   ├── batch_size: 16
│   │   │   │   └── stop_criteria:
│   │   │   │       ├── train_loss: 3.0
│   │   │   │       ├── min_epochs: 500
│   │   │   │       └── max_epochs: 10000
│   │   │   └── AUTHORS (8 total)
│   │   │
│   │   └── experiment.py            [EXPERIMENT CONFIG]
│   │       └── Experiment class
│   │           ├── train_author
│   │           ├── seed
│   │           ├── eval_paths
│   │           ├── stop_criteria
│   │           └── resume_training
│   │
│   ├── data/
│   │   └── loader.py                [DATA LOADING]
│   │       ├── tokenize_texts()
│   │       ├── sample_tokens()
│   │       ├── OnTheFlyTrainingDataset
│   │       ├── EvalDataset
│   │       ├── get_train_data_loader()
│   │       └── get_eval_data_loader()
│   │
│   ├── models/
│   │   └── __init__.py              (empty - model classes imported from HF)
│   │
│   └── [visualization, analysis, etc.]
│
├── data/
│   ├── cleaned/
│   │   ├── baum/                    Training texts
│   │   ├── thompson/
│   │   ├── austen/
│   │   ├── [etc.]
│   │   ├── content_only/            Variant: content words only
│   │   ├── function_only/           Variant: function words only
│   │   └── pos_only/                Variant: POS tags only
│   │
│   ├── raw/                         Original texts
│   └── book_and_chapter_titles.txt
│
└── models/
    ├── {author}_tokenizer=gpt2_seed=0/       Baseline models
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── model.safetensors
    │   ├── training_state.pt                 [CRITICAL FOR RESUME]
    │   └── loss_logs.csv
    │
    └── {author}_variant=content_tokenizer=gpt2_seed=0/  Variant models
        ├── [same structure as baseline]
        └── loss_logs.csv
```

---

## Call Graph: Training Pipeline

```
code/main.py
  │
  ├─ if __name__ == "__main__":
  │   │
  │   ├─ 1. Load environment
  │   │   ├─ RESUME_TRAINING env var
  │   │   ├─ ANALYSIS_VARIANT env var
  │   │   └─ get_device_info()
  │   │
  │   ├─ 2. Create 80 experiments
  │   │   └─ Experiment(author, seed) × 8 authors × 10 seeds
  │   │
  │   ├─ 3. Filter based on resume mode
  │   │   └─ check_model_complete(exp.name)
  │   │       ├─ Returns (is_complete, has_weights, epochs_done)
  │   │       └─ Decision: skip, resume, or start fresh
  │   │
  │   └─ 4. Submit training jobs
  │       └─ if multiprocessing:
  │           └─ pool.apply_async(run_experiment, ...)
  │       └─ else:
  │           └─ for exp in experiments: run_experiment(exp)
  │
  └─ run_experiment(exp, device_queue, device_type)
      │
      ├─ 1. Setup
      │   ├─ Get device (cuda/mps/cpu)
      │   └─ Set random seeds
      │
      ├─ 2. Load tokenizer
      │   └─ get_tokenizer(exp.tokenizer_name)
      │
      ├─ 3. Load/init model
      │   └─ if exp.resume_training:
      │       └─ load_checkpoint(GPT2LMHeadModel, exp.name, device)
      │           ├─ model.from_pretrained(checkpoint_dir)
      │           ├─ optimizer.load_state_dict()
      │           ├─ random states restored
      │           └─ return (model, optimizer, start_epoch)
      │   └─ else:
      │       └─ init_model(GPT2LMHeadModel, exp.name, device, exp.lr, config)
      │           ├─ GPT2LMHeadModel(config).to(device)
      │           ├─ AdamW(model.parameters(), lr)
      │           └─ return (model, optimizer)
      │
      ├─ 4. Create dataloaders
      │   ├─ train_dataloader = get_train_data_loader(...)
      │   │   ├─ tokenize_texts()
      │   │   ├─ sample_tokens()
      │   │   ├─ OnTheFlyTrainingDataset()
      │   │   └─ DataLoader(shuffle=True)
      │   │
      │   └─ eval_dataloaders = {...}
      │       └─ get_eval_data_loader() for each author
      │           ├─ tokenize_texts()
      │           ├─ EvalDataset()
      │           └─ DataLoader(shuffle=False)
      │
      ├─ 5. Initial evaluation
      │   └─ for name, eval_dataloader in eval_dataloaders:
      │       └─ evaluate_model()
      │
      ├─ 6. MAIN TRAINING LOOP
      │   └─ for epoch in range(start_epoch, max_epochs):
      │       │
      │       ├─ Training phase
      │       │   └─ for batch in train_dataloader:
      │       │       ├─ model.train()
      │       │       ├─ Forward: outputs = model(input_ids)
      │       │       ├─ Backward: optimizer.step()
      │       │       ├─ Accumulate loss
      │       │       └─ Memory cleanup
      │       │
      │       ├─ Evaluation phase
      │       │   ├─ avg_train_loss = total / len(dataloader)
      │       │   │
      │       │   └─ for name, eval_dataloader:
      │       │       ├─ evaluate_model() -> eval_loss
      │       │       └─ update_loss_log()
      │       │
      │       ├─ Checkpoint
      │       │   └─ save_checkpoint(model, optimizer, exp.name, epoch+1)
      │       │       ├─ model.save_pretrained()
      │       │       └─ torch.save(training_state, ...)
      │       │
      │       └─ Check stopping criteria
      │           └─ if train_loss <= 3.0 AND epoch >= 500:
      │               └─ break
      │
      └─ 7. Cleanup
          └─ Return GPU to queue (if multiprocessing)
```

---

## File Dependencies

### Direct Training Loop Dependencies

```
main.py
├─ imports from code/:
│  ├─ model_utils.py (save_checkpoint, load_checkpoint, init_model)
│  ├─ data_utils.py (get_train_data_loader, get_eval_data_loader)
│  ├─ eval_utils.py (evaluate_model)
│  ├─ logging_utils.py (update_loss_log)
│  ├─ tokenizer_utils.py (get_tokenizer)
│  ├─ experiment.py (Experiment class)
│  └─ constants.py (MODELS_DIR, AUTHORS, etc.)
│
└─ imports from transformers:
   ├─ GPT2Config
   ├─ GPT2LMHeadModel
   └─ AdamW (from torch.optim)
```

### Model/Checkpoint Dependencies

```
model_utils.py
├─ torch
├─ transformers.GPT2LMHeadModel (implicitly, via parameter)
├─ torch.optim.AdamW
└─ constants.py (MODELS_DIR)
```

### Data Loading Dependencies

```
llm_stylometry/data/loader.py
├─ torch
├─ pathlib.Path
└─ core/constants.py (CLEANED_DATA_DIR)
```

### Configuration Dependencies

```
llm_stylometry/core/constants.py
├─ pathlib.Path
└─ DEFAULT_CONFIG (dict)

llm_stylometry/core/experiment.py
├─ core/constants.py (DEFAULT_CONFIG, AUTHORS, CLEANED_DATA_DIR)
└─ pathlib.Path
```

---

## Critical Files by Function

### 1. TRAINING LOOP
- **Primary:** `/code/main.py` (lines 254-352)
  - Where `model.train()`, `optimizer.step()` happen
  - Where stop criteria are checked

### 2. CHECKPOINTING
- **Save:** `/code/model_utils.py` save_checkpoint() (lines 12-39)
- **Load:** `/code/model_utils.py` load_checkpoint() (lines 42-98)
- **Init:** `/code/model_utils.py` init_model() (lines 101-112)

### 3. STOP CRITERIA
- **Definition:** `/llm_stylometry/core/constants.py` (lines 53-57)
  - stop_criteria dict with train_loss, min_epochs, max_epochs
- **Enforcement:** `/code/main.py` (lines 348-352)
  - if train_loss <= 3.0 and epochs >= 500: break

### 4. DATA LOADING
- **Training:** `/llm_stylometry/data/loader.py` get_train_data_loader()
- **Evaluation:** `/llm_stylometry/data/loader.py` get_eval_data_loader()
- **Tokenization:** `/llm_stylometry/data/loader.py` tokenize_texts()
- **Sampling:** `/llm_stylometry/data/loader.py` sample_tokens()

### 5. EXPERIMENT CONFIG
- **Definition:** `/llm_stylometry/core/experiment.py` Experiment class
- **Instantiation:** `/code/main.py` lines 119-129
- **Usage:** Passed to `run_experiment(exp, ...)`

### 6. EVALUATION
- **Eval Loop:** `/code/eval_utils.py` evaluate_model()
- **Logging:** `/code/logging_utils.py` update_loss_log()

---

## Key Variable Flow

```
Constants (core/constants.py, code/constants.py)
  ├─ DEFAULT_CONFIG
  │  ├─ n_positions: 1024
  │  ├─ n_embd: 128
  │  ├─ batch_size: 16
  │  ├─ lr: 5e-5
  │  └─ stop_criteria: {train_loss: 3.0, min_epochs: 500, max_epochs: 10000}
  │
  └─ AUTHORS, MODELS_DIR, CLEANED_DATA_DIR

        ↓

Experiment (per author × seed)
  ├─ train_author (e.g., "baum")
  ├─ seed (0-9)
  ├─ eval_paths (dict of author → book path)
  ├─ stop_criteria (from DEFAULT_CONFIG)
  ├─ resume_training (from env var)
  └─ [all hyperparams from DEFAULT_CONFIG]

        ↓

run_experiment(exp)
  ├─ Model: GPT2LMHeadModel(config)
  ├─ Optimizer: AdamW(model.parameters(), lr)
  ├─ Data: get_train_data_loader(exp.train_author, ...)
  │        get_eval_data_loader(exp.eval_paths, ...)
  │
  ├─ Training Loop
  │  ├─ for epoch in range(start_epoch, max_epochs)
  │  ├─ for batch in train_dataloader
  │  │  └─ optimizer.step()
  │  │
  │  └─ Check: train_loss <= exp.stop_criteria["train_loss"]
  │            AND epochs >= exp.stop_criteria["min_epochs"]
  │
  └─ save_checkpoint()
     └─ Save training_state.pt (with optimizer state for resume)
```

---

## Resume Training Flow

```
RESUME_TRAINING=1 python code/main.py
        ↓
check_model_complete(model_name)
        ├─ Read loss_logs.csv
        ├─ Get last_epoch, last_train_loss
        │
        └─ Return (is_complete, has_weights, epochs_done)
            ├─ is_complete: loss <= 3.0 AND epoch >= 500
            ├─ has_weights: config.json + model.safetensors exist
            └─ epochs_done: epochs completed so far

        ↓

if is_complete:
    skip (training already done)
    
elif has_weights:
    load_checkpoint()
        ├─ model.from_pretrained(checkpoint_dir)
        ├─ Load training_state.pt
        ├─ Restore optimizer state
        ├─ Restore all RNG states
        └─ return (model, optimizer, start_epoch)
    
    # Continue from start_epoch
    for epoch in range(start_epoch, max_epochs):
        ...
    
else:
    # No weights - start fresh
    init_model()
    # Remove loss_logs.csv if exists
    for epoch in range(0, max_epochs):
        ...
```

---

Generated: 2025-10-24
