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

# Analysis variants
ANALYSIS_VARIANTS = ['content', 'function', 'pos']


def get_data_dir(variant=None):
    """
    Get data directory based on analysis variant.

    Args:
        variant: One of ANALYSIS_VARIANTS or None for baseline

    Returns:
        Path to data directory
    """
    if variant is None:
        return CLEANED_DATA_DIR

    if variant not in ANALYSIS_VARIANTS:
        raise ValueError(f"Invalid variant: {variant}. Must be one of {ANALYSIS_VARIANTS}")

    variant_dir = CLEANED_DATA_DIR / f"{variant}_only"
    if not variant_dir.exists():
        raise FileNotFoundError(f"Variant directory not found: {variant_dir}")

    return variant_dir
