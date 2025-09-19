import torch
import logging
import sys
import warnings
from transformers import GPT2Config, GPT2LMHeadModel
from data_utils import get_train_data_loader, get_eval_data_loader

# Suppress the loss_type warning from transformers
warnings.filterwarnings("ignore", message=".*loss_type.*unrecognized.*")
from model_utils import (
    save_checkpoint,
    load_checkpoint,
    init_model,
    count_non_embedding_params,
)
from tokenizer_utils import get_tokenizer
from eval_utils import evaluate_model
from logging_utils import update_loss_log
import random
import numpy as np
import torch.backends.cudnn as cudnn
from experiment import Experiment
import torch.multiprocessing as mp
from constants import MODELS_DIR, AUTHORS, CLEANED_DATA_DIR
import os

# Disable tqdm if running in subprocess or if explicitly disabled
USE_TQDM = os.environ.get('DISABLE_TQDM', '0') != '1' and sys.stdout.isatty()
if USE_TQDM:
    from tqdm import tqdm
else:
    # Simple replacement that just returns the iterable
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_complete(model_name, stop_train_loss=3.0, min_epochs=0):
    """
    Check if a model has completed training based on loss logs and weights.

    Returns:
        tuple: (is_complete, has_weights, epochs_completed)
            - is_complete: True if model has met stop criteria
            - has_weights: True if model weights exist
            - epochs_completed: Number of epochs completed (0 if no logs)
    """
    model_dir = MODELS_DIR / model_name

    # Check if model weights exist
    weights_file = model_dir / "model.safetensors"
    config_file = model_dir / "config.json"
    training_state_file = model_dir / "training_state.pt"
    has_weights = weights_file.exists() and config_file.exists() and training_state_file.exists()

    # Check loss logs
    loss_log_file = model_dir / "loss_logs.csv"
    if not loss_log_file.exists():
        return False, has_weights, 0

    # Read loss logs to check training status
    import pandas as pd
    try:
        df = pd.read_csv(loss_log_file)
        if df.empty:
            return False, has_weights, 0

        # Get the last training loss for this model
        train_losses = df[df['loss_dataset'] == 'train'].sort_values('epochs_completed')
        if train_losses.empty:
            return False, has_weights, 0

        last_epoch = train_losses['epochs_completed'].max()
        last_train_loss = train_losses[train_losses['epochs_completed'] == last_epoch]['loss_value'].iloc[0]

        # Check if model has met stop criteria
        is_complete = (last_train_loss <= stop_train_loss and last_epoch >= min_epochs)

        return is_complete, has_weights, int(last_epoch)
    except Exception as e:
        logger.warning(f"Error reading loss logs for {model_name}: {e}")
        return False, has_weights, 0

# Detect available devices
def get_device_info():
    """Detect and return device configuration."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        return "cuda", device_count
    elif torch.backends.mps.is_available():
        # Apple Metal Performance Shaders (MPS) backend
        return "mps", 1
    else:
        return "cpu", 1

device_type, device_count = get_device_info()
logger.info(f"Device type: {device_type}, Count: {device_count}")

# Check if we're in resume mode
resume_mode = os.environ.get('RESUME_TRAINING', '0') == '1'

experiments = []
for seed in range(10):
    for author in AUTHORS:
        experiments.append(
            Experiment(
                train_author=author,
                seed=seed,
                tokenizer_name="gpt2",
                resume_training=resume_mode,
            )
        )


def run_experiment(exp: Experiment, device_queue, device_type="cuda"):
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Get an available device id
        device_id = device_queue.get() if device_queue else 0
        logger.info(f"Starting experiment: {exp.name}")

        # Set up device based on type
        if device_type == "cuda":
            torch.cuda.set_device(device_id)
            device = torch.device("cuda", index=device_id)
            device_label = f"GPU {device_id}"
        elif device_type == "mps":
            device = torch.device("mps")
            device_label = "MPS"
        else:
            device = torch.device("cpu")
            device_label = "CPU"

        # Initialize tokenizer directly using get_tokenizer
        tokenizer = get_tokenizer(exp.tokenizer_name)

        # Set random seeds for reproducibility
        assert exp.seed is not None
        random.seed(exp.seed)
        np.random.seed(exp.seed)
        torch.manual_seed(exp.seed)

        # Set up train dataloader with on-the-fly sampling
        train_dataloader = get_train_data_loader(
            path=CLEANED_DATA_DIR / exp.train_author,
            tokenizer=tokenizer,
            n_positions=exp.n_positions,
            batch_size=exp.batch_size,
            n_tokens=exp.n_train_tokens,
            seed=exp.seed,
            excluded_train_path=exp.excluded_train_path,
        )
        logger.info(
            f"[{device_label}] Number of training batches: {len(train_dataloader)}"
        )

        # Set up eval dataloaders
        eval_dataloaders = {
            name: get_eval_data_loader(
                path=path,
                tokenizer=tokenizer,
                n_positions=exp.n_positions,
                batch_size=exp.batch_size,
            )
            for name, path in exp.eval_paths.items()
        }

        # Set up model configuration
        config = GPT2Config(
            n_positions=exp.n_positions,
            n_embd=exp.n_embd,
            n_layer=exp.n_layer,
            n_head=exp.n_head,
        )
        modelname = exp.name

        stop_train_loss = exp.stop_criteria["train_loss"]
        min_epochs = exp.stop_criteria["min_epochs"]
        max_epochs = exp.stop_criteria["max_epochs"]
        assert stop_train_loss is not None
        assert min_epochs is not None
        assert max_epochs is not None

        # Logic for loading or initializing the model
        if exp.resume_training:
            model, optimizer, start_epoch = load_checkpoint(
                model_class=GPT2LMHeadModel,
                model_name=modelname,
                device=device,
            )
        else:
            model, optimizer = init_model(
                model_class=GPT2LMHeadModel,
                model_name=modelname,
                config=config,
                device=device,
                lr=exp.lr,
            )
            start_epoch = 0

        logger.info(
            f"[{device_label}] Total number of non-embedding parameters: {count_non_embedding_params(model)}"
        )

        # Initial evaluation (epochs_complete = 0)
        for name, eval_dataloader in eval_dataloaders.items():
            eval_loss = evaluate_model(
                model=model,
                eval_dataloader=eval_dataloader,
                device=device,
            )

            # Log each evaluation separately
            update_loss_log(
                log_file_path=MODELS_DIR / modelname / "loss_logs.csv",
                epochs_completed=0,
                loss_dataset=name,
                loss_value=eval_loss,
                seed=exp.seed,
                train_author=exp.train_author,
            )

        # Set up mixed precision training if supported
        use_amp = device_type == "cuda"
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Enable gradient checkpointing to save memory (if supported)
        try:
            model.gradient_checkpointing_enable()
            logger.info(f"[{device_label}] Gradient checkpointing enabled for memory efficiency")
        except AttributeError:
            logger.info(f"[{device_label}] Model does not support gradient checkpointing")

        # Training loop
        for epoch in tqdm(range(start_epoch, max_epochs)):
            total_train_loss = 0.0

            # Iterate over batches in the training dataloader
            for batch_idx, batch in enumerate(train_dataloader):
                model.train()

                input_ids = batch["input_ids"].to(device)

                # Forward pass with or without mixed precision
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, labels=input_ids)
                        loss = outputs.loss
                else:
                    outputs = model(input_ids=input_ids, labels=input_ids)
                    loss = outputs.loss

                # Backward pass with or without mixed precision
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Accumulate training loss
                total_train_loss += loss.item()

                # Delete intermediate tensors to free memory
                del outputs, loss

                # Clear CUDA cache periodically
                if (batch_idx + 1) % 5 == 0:
                    torch.cuda.empty_cache()

            epochs_completed = epoch + 1

            # Calculate average training loss
            train_loss = total_train_loss / len(train_dataloader)

            # Log training loss from the current epoch
            update_loss_log(
                log_file_path=MODELS_DIR / modelname / "loss_logs.csv",
                train_author=exp.train_author,
                loss_dataset="train",
                loss_value=train_loss,
                epochs_completed=epochs_completed,
                seed=exp.seed,
            )

            # Evaluate and log each eval dataset separately
            eval_losses = {}  # For log message
            for name, eval_dataloader in eval_dataloaders.items():
                # Evaluate one dataset
                eval_loss = evaluate_model(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    device=device,
                )

                # Store for later console output
                eval_losses[name] = eval_loss

                # Log each evaluation result
                update_loss_log(
                    log_file_path=MODELS_DIR / modelname / "loss_logs.csv",
                    epochs_completed=epochs_completed,
                    loss_dataset=name,
                    loss_value=eval_loss,
                    seed=exp.seed,
                    train_author=exp.train_author,
                )

                # Force memory cleanup between evaluations (CUDA only)
                if device_type == "cuda":
                    torch.cuda.empty_cache()

            # Build log message for console output
            log_message = f"[{device_label}] Epoch {epochs_completed}/{max_epochs}: training loss = {train_loss:.4f}"
            for name, loss in eval_losses.items():
                log_message += f", {name}: {loss:.4f}"
            logger.info(log_message)

            # Save the model checkpoint at the end of each epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                model_name=modelname,
                epochs_completed=epochs_completed,
            )
            # Early stopping after completing epoch (retain logs and checkpoints)
            if train_loss <= stop_train_loss and min_epochs <= epochs_completed:
                logger.info(
                    f"[{device_label}] Training loss {train_loss:.4f} below threshold {stop_train_loss}. Stopping training."
                )
                break
        logger.info(f"[{device_label}] Training complete for {modelname}")

        # Return the GPU id to the queue
        if device_queue:
            device_queue.put(device_id)
    except Exception:
        logger.exception(f"Error in experiment {exp.name}")
        raise


if __name__ == "__main__":
    # Check if we should run sequentially (for subprocess compatibility)
    USE_MULTIPROCESSING = os.environ.get('NO_MULTIPROCESSING', '0') != '1'

    # Filter experiments based on resume mode
    if resume_mode:
        logger.info("Checking existing models for resume...")
        experiments_to_run = []
        import shutil

        for exp in experiments:
            is_complete, has_weights, epochs_done = check_model_complete(
                exp.name,
                exp.stop_criteria["train_loss"],
                exp.stop_criteria["min_epochs"]
            )

            if is_complete:
                # Model has completed training - skip it
                logger.info(f"Skipping {exp.name} - already complete (epochs: {epochs_done})")
            elif has_weights:
                # Model has weights and can be resumed
                logger.info(f"Resuming {exp.name} from epoch {epochs_done}")
                experiments_to_run.append(exp)
            elif epochs_done > 0:
                # Loss logs exist but no weights (e.g., after cloning repo) - need to restart
                logger.info(f"Starting {exp.name} from scratch - no weights available (removing existing logs)")
                model_dir = MODELS_DIR / exp.name
                if model_dir.exists():
                    # Remove only this specific model's directory to start fresh
                    shutil.rmtree(model_dir)
                exp.resume_training = False  # Force fresh start for this model
                experiments_to_run.append(exp)
            else:
                # No logs or weights - start fresh for this model
                logger.info(f"Starting fresh: {exp.name} (no existing logs or weights)")
                exp.resume_training = False  # No checkpoint to resume from
                experiments_to_run.append(exp)

        experiments = experiments_to_run
        total_models = 80  # 8 authors × 10 seeds
        logger.info(f"Models to train: {len(experiments)} out of {total_models} total")

        if not experiments:
            logger.info("All models are complete. Nothing to train.")
            sys.exit(0)

    # Use already detected device configuration
    if device_type == "cuda":
        # Check for MAX_GPUS environment variable to optionally limit GPU usage
        max_gpus = int(os.environ.get('MAX_GPUS', '0')) or device_count
        gpu_count = min(device_count, max_gpus)
        if gpu_count < device_count:
            print(f"Using {gpu_count} GPUs (limited by MAX_GPUS) out of {device_count} available")
        else:
            print(f"Using all {gpu_count} available GPUs")
    elif device_type == "mps":
        gpu_count = 1
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        gpu_count = 1
        print("Using CPU for training (this will be slow)")

    if USE_MULTIPROCESSING and device_type == "cuda" and gpu_count > 1:
        # Only use multiprocessing for multiple CUDA GPUs
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        device_queue = manager.Queue()
        for gpu in range(gpu_count):
            device_queue.put(gpu)

        pool = mp.Pool(processes=gpu_count)
        logger = logging.getLogger(__name__)

        def error_callback(e):
            logger.exception("Unhandled error in worker, shutting down all processes")
            pool.terminate()
            sys.exit(1)

        for exp in experiments:
            pool.apply_async(
                run_experiment, (exp, device_queue, device_type), error_callback=error_callback
            )
        pool.close()
        pool.join()
    else:
        # Sequential mode for subprocess compatibility or single device
        print("Running in sequential mode (multiprocessing disabled)")
        if device_type == "cuda" and gpu_count > 1:
            # Multiple GPUs but running sequentially
            import queue
            device_queue = queue.Queue()
            for gpu in range(gpu_count):
                device_queue.put(gpu)
        else:
            # Single device or non-CUDA
            device_queue = None

        for i, exp in enumerate(experiments):
            print(f"Training model {i+1}/{len(experiments)}: {exp.name}")
            run_experiment(exp, device_queue, device_type)
            # For multi-GPU sequential mode, rotate through GPUs
            if device_queue and not device_queue.empty():
                device_id = device_queue.get()
                device_queue.put(device_id)
