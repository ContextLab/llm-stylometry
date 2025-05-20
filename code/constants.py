from pathlib import Path


def find_project_root():
    current_path = Path(__file__).resolve().parent
    while current_path != current_path.parent:
        if current_path.name == "llm-stylometry":
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root not found.")


# Base paths
ROOT_DIR = find_project_root()
CODE_DIR = ROOT_DIR / "code"
DATA_DIR = ROOT_DIR / "data"

# Data-related paths
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
TITLES_FILE = DATA_DIR / "book_and_chapter_titles.txt"

# Models paths
MODELS_DIR = ROOT_DIR / "models"

# Figures paths
FIGURES_DIR = ROOT_DIR / "figures"

# Authors
AUTHORS = [
    "baum",
    "thompson",
    "dickens",
    "melville",
    "wells",
    "austen",
    "fitzgerald",
    "twain",
]
