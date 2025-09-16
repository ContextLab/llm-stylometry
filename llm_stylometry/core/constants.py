"""Constants and configuration for the llm-stylometry package."""

from pathlib import Path


def find_project_root():
    """Find the project root directory."""
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:
        if current_path.name == "llm-stylometry":
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found.")


# Base paths
ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "data"

# Data-related paths
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
TITLES_FILE = DATA_DIR / "book_and_chapter_titles.txt"

# Models paths
MODELS_DIR = ROOT_DIR / "models"

# Paper paths
PAPER_DIR = ROOT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figs" / "source"

# Authors - standardized order for all visualizations
AUTHORS = [
    "baum",
    "thompson",
    "austen",
    "dickens",
    "fitzgerald",
    "melville",
    "twain",
    "wells",
]

# Model configuration defaults
DEFAULT_CONFIG = {
    "n_positions": 1024,
    "n_embd": 128,
    "n_layer": 8,
    "n_head": 8,
    "batch_size": 16,
    "lr": 5e-5,
    "n_train_tokens": 643041,
    "stop_criteria": {
        "train_loss": 3.0,
        "min_epochs": 500,
        "max_epochs": 10000,
    }
}