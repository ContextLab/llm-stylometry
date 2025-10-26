# Training Code Snippets - Quick Reference

## 1. The Complete Training Loop (Simplified)

**From: `/code/main.py`, Lines 254-352**

```python
# Setup
model = GPT2LMHeadModel(config).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
train_dataloader = get_train_data_loader(...)
eval_dataloaders = {...}

# Training loop
for epoch in range(start_epoch, max_epochs):
    total_train_loss = 0.0
    
    # Training phase
    for batch in train_dataloader:
        model.train()
        input_ids = batch["input_ids"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # Epoch summary
    train_loss = total_train_loss / len(train_dataloader)
    
    # Evaluation phase
    for name, eval_dataloader in eval_dataloaders.items():
        eval_loss = evaluate_model(model, eval_dataloader, device)
        log_loss(name, eval_loss, epoch)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, model_name, epoch+1)
    
    # Check stopping criteria
    if train_loss <= 3.0 and (epoch+1) >= 500:
        break
```

---

## 2. Checkpoint Saving

**From: `/code/model_utils.py`, Lines 12-39**

```python
def save_checkpoint(model, optimizer, model_name, epochs_completed):
    checkpoint_dir = MODELS_DIR / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights (HF format)
    model.save_pretrained(save_directory=checkpoint_dir)
    
    # Save training state for resume
    training_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs_completed": epochs_completed,
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    torch.save(training_state, checkpoint_dir / "training_state.pt")
    logger.info(f"Checkpoint saved at epoch {epochs_completed}")
```

---

## 3. Checkpoint Loading (Resume)

**From: `/code/model_utils.py`, Lines 42-98**

```python
def load_checkpoint(model_class, model_name, device):
    checkpoint_dir = MODELS_DIR / model_name
    
    # Load model weights
    model = model_class.from_pretrained(checkpoint_dir).to(device)
    
    # Load training state
    training_state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location=device
    )
    
    # Restore optimizer
    optimizer = AdamW(model.parameters(), lr=0)
    optimizer.load_state_dict(training_state["optimizer_state_dict"])
    
    # Restore RNG states
    random.setstate(training_state["random_state"])
    np.random.set_state(training_state["np_random_state"])
    torch.set_rng_state(training_state["torch_random_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(training_state["cuda_random_state"])
    
    epochs_completed = training_state["epochs_completed"]
    
    return model, optimizer, epochs_completed
```

---

## 4. Model Initialization

**From: `/code/model_utils.py`, Lines 101-112**

```python
def init_model(model_class, model_name, device, lr, config):
    # Create model from config
    model = model_class(config=config).to(device)
    
    # Create optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)
    
    # Create model directory
    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    return model, optimizer
```

**Config parameters:**
```python
config = GPT2Config(
    n_positions=1024,   # Context length
    n_embd=128,         # Embedding dim
    n_layer=8,          # Number of transformer layers
    n_head=8,           # Number of attention heads
)
```

---

## 5. Stop Criteria Check

**From: `/code/main.py`, Lines 348-352**

```python
# After each epoch:
if train_loss <= stop_train_loss and min_epochs <= epochs_completed:
    logger.info(
        f"Training loss {train_loss:.4f} below threshold {stop_train_loss}. "
        f"Stopping training."
    )
    break
```

**Parameters:**
```python
stop_train_loss = exp.stop_criteria["train_loss"]      # 3.0
min_epochs = exp.stop_criteria["min_epochs"]           # 500
max_epochs = exp.stop_criteria["max_epochs"]           # 10,000
```

---

## 6. Training Data Loader

**From: `/llm_stylometry/data/loader.py`, Lines 236-277**

```python
def get_train_data_loader(path, tokenizer, n_positions, batch_size, n_tokens, seed, excluded_train_path=None):
    # Load and tokenize texts
    tokenized_texts = tokenize_texts(tokenizer, path, excluded_train_path)
    
    # Sample tokens proportionally
    sampled_tokens = sample_tokens(tokenized_texts, n_tokens, seed)
    
    # Create dataset (on-the-fly random sampling)
    dataset = OnTheFlyTrainingDataset(sampled_tokens, n_positions)
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
```

---

## 7. Evaluation Data Loader

**From: `/llm_stylometry/data/loader.py`, Lines 280-327**

```python
def get_eval_data_loader(path, tokenizer, n_positions, batch_size):
    # Load and tokenize evaluation text
    text = path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    
    # Create sequential evaluation dataset
    dataset = EvalDataset(tokens, n_positions, tokenizer)
    
    # Create dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential for evaluation
    )
```

---

## 8. Evaluate Model (Per Epoch)

**From: `/code/eval_utils.py`, Lines 8-37**

```python
def evaluate_model(model, eval_dataloader, device):
    """Returns average loss across evaluation dataset"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            total_loss += loss
    
    return total_loss / len(eval_dataloader)
```

---

## 9. Resume Training Logic

**From: `/code/main.py`, Lines 368-408**

```python
def check_model_complete(model_name, stop_train_loss=3.0, min_epochs=500):
    """Returns (is_complete, has_weights, epochs_completed)"""
    model_dir = MODELS_DIR / model_name
    
    # Check if weights exist
    has_weights = (
        (model_dir / "model.safetensors").exists() and
        (model_dir / "config.json").exists() and
        (model_dir / "training_state.pt").exists()
    )
    
    # Check loss logs
    loss_log_file = model_dir / "loss_logs.csv"
    if not loss_log_file.exists():
        return False, has_weights, 0
    
    df = pd.read_csv(loss_log_file)
    train_losses = df[df['loss_dataset'] == 'train'].sort_values('epochs_completed')
    
    last_epoch = train_losses['epochs_completed'].max()
    last_train_loss = train_losses[train_losses['epochs_completed'] == last_epoch]['loss_value'].iloc[0]
    
    is_complete = (last_train_loss <= stop_train_loss and last_epoch >= min_epochs)
    
    return is_complete, has_weights, int(last_epoch)
```

---

## 10. Multiprocessing Setup

**From: `/code/main.py`, Lines 426-447**

```python
mp.set_start_method("spawn", force=True)
manager = mp.Manager()
device_queue = manager.Queue()

# Add GPUs to queue
for gpu in range(gpu_count):
    device_queue.put(gpu)

# Create pool and submit jobs
pool = mp.Pool(processes=gpu_count)

for exp in experiments:
    pool.apply_async(
        run_experiment, 
        (exp, device_queue, device_type),
        error_callback=error_callback
    )

pool.close()
pool.join()
```

---

## 11. Mixed Precision Training (CUDA)

**From: `/code/main.py`, Lines 243-277**

```python
use_amp = device_type == "cuda"
scaler = torch.amp.GradScaler('cuda') if use_amp else None

# In training loop:
if use_amp:
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 12. Device Detection

**From: `/code/main.py`, Lines 86-98**

```python
def get_device_info():
    """Detect available devices"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        return "cuda", device_count
    elif torch.backends.mps.is_available():
        return "mps", 1
    else:
        return "cpu", 1

device_type, device_count = get_device_info()

# In training:
if device_type == "cuda":
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", index=device_id)
elif device_type == "mps":
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

---

## 13. Experiment Creation

**From: `/code/main.py`, Lines 119-129**

```python
from experiment import Experiment

experiments = []
for seed in range(10):
    for author in AUTHORS:
        experiments.append(
            Experiment(
                train_author=author,
                seed=seed,
                tokenizer_name="gpt2",
                analysis_variant=variant,  # None, 'content', 'function', 'pos'
                resume_training=resume_mode,
            )
        )
# Creates 80 models (8 authors × 10 seeds)
```

---

## Key Variables & Constants

```python
# Training hyperparameters
n_positions = 1024           # Context window size
batch_size = 16              # Training batch size
n_train_tokens = 643041      # Tokens per author
lr = 5e-5                    # Learning rate

# Stop criteria
stop_train_loss = 3.0        # Training loss threshold
min_epochs = 500             # Minimum epochs before stopping allowed
max_epochs = 10000           # Hard limit on epochs

# Authors (8 total)
AUTHORS = ['baum', 'thompson', 'austen', 'dickens', 
           'fitzgerald', 'melville', 'twain', 'wells']

# Data directories
MODELS_DIR = Path("models")
CLEANED_DATA_DIR = Path("data/cleaned")

# Evaluation sets (per author)
eval_paths = {
    'baum': Path("data/cleaned/baum/..."),      # 1 random book
    'thompson': Path("data/cleaned/thompson/..."),
    # ... one for each author
    'non_oz_baum': Path("data/cleaned/non_oz_baum/..."),      # Special
    'non_oz_thompson': Path("data/cleaned/non_oz_thompson/..."),
    'contested': Path("data/cleaned/contested/..."),
}
```

---

## Environment Variables

```bash
# Resume training
RESUME_TRAINING=1 python code/main.py

# Limit GPU usage
MAX_GPUS=2 python code/main.py

# Run analysis variant
ANALYSIS_VARIANT=content python code/main.py
ANALYSIS_VARIANT=function python code/main.py
ANALYSIS_VARIANT=pos python code/main.py

# Disable multiprocessing
NO_MULTIPROCESSING=1 python code/main.py

# Disable tqdm
DISABLE_TQDM=1 python code/main.py
```

---

Generated: 2025-10-24
