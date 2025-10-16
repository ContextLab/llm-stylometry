import torch
from pathlib import Path
import logging
from torch.optim import AdamW
from constants import MODELS_DIR
import random
import numpy as np

logger = logging.getLogger(__name__)


def save_checkpoint(
    model,
    optimizer,
    model_name,
    epochs_completed,
):
    checkpoint_dir = MODELS_DIR / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_directory=checkpoint_dir)

    # Save training state including random states for deterministic resume
    training_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs_completed": epochs_completed,
        "random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    # Also save CUDA random state if available
    if torch.cuda.is_available():
        training_state["cuda_random_state"] = torch.cuda.get_rng_state_all()

    torch.save(obj=training_state, f=checkpoint_dir / "training_state.pt")
    logger.info(
        f"Checkpoint saved for {model_name} at epochs_completed={epochs_completed}"
    )


def load_checkpoint(model_class, model_name, device):
    checkpoint_dir = MODELS_DIR / model_name

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found for {model_name}")

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=checkpoint_dir
    ).to(device)

    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        raise FileNotFoundError(f"Training state file not found for {model_name}")

    training_state = torch.load(f=training_state_path, map_location=device)

    optimizer = AdamW(params=model.parameters(), lr=0)
    optimizer.load_state_dict(state_dict=training_state["optimizer_state_dict"])
    epochs_completed = training_state["epochs_completed"]

    # Restore random states for deterministic resume (if available)
    if "random_state" in training_state:
        random.setstate(training_state["random_state"])
        logger.info("Restored Python random state")

    if "np_random_state" in training_state:
        np.random.set_state(training_state["np_random_state"])
        logger.info("Restored NumPy random state")

    if "torch_random_state" in training_state:
        try:
            # torch.set_rng_state() requires CPU tensor
            rng_state = training_state["torch_random_state"]
            if rng_state.device.type != 'cpu':
                rng_state = rng_state.cpu()
            torch.set_rng_state(rng_state)
            logger.info("Restored PyTorch random state")
        except Exception as e:
            logger.warning(f"Could not restore PyTorch RNG state: {e}. Continuing with random initialization.")

    if "cuda_random_state" in training_state and torch.cuda.is_available():
        try:
            # Ensure CUDA RNG states are on correct devices
            cuda_states = training_state["cuda_random_state"]
            if isinstance(cuda_states, list):
                # Move each state to CPU if needed (set_rng_state_all handles device placement)
                cuda_states = [s.cpu() if hasattr(s, 'cpu') and s.device.type != 'cpu' else s for s in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)
            logger.info("Restored CUDA random state")
        except Exception as e:
            logger.warning(f"Could not restore CUDA RNG state: {e}. Continuing with random initialization.")

    logger.info(
        f"Checkpoint loaded for {model_name} from epochs_completed={epochs_completed}"
    )

    return model, optimizer, epochs_completed


def init_model(model_class, model_name, device, lr, config):
    model = model_class(config=config).to(device)
    logger.info("Initialized new model")

    optimizer = AdamW(params=model.parameters(), lr=lr)
    logger.info("Initialized new optimizer")

    model_dir = MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory created at {model_dir}")

    return model, optimizer


def count_non_embedding_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "wte" in name or "wpe" in name
    )
    non_embedding_params = total_params - embedding_params
    return non_embedding_params
