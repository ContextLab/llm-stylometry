import torch
from pathlib import Path
import logging
from torch.optim import AdamW
from constants import MODELS_DIR

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

    training_state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs_completed": epochs_completed,
    }
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

    training_state = torch.load(f=training_state_path)

    optimizer = AdamW(params=model.parameters(), lr=0)
    optimizer.load_state_dict(state_dict=training_state["optimizer_state_dict"])
    epochs_completed = training_state["epochs_completed"]
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
