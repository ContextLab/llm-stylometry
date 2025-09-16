#!/usr/bin/env python
"""Create small test dataset for quick testing."""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Create test directory
test_dir = Path(__file__).parent
data_dir = test_dir / "data"
data_dir.mkdir(exist_ok=True)

# Create small synthetic model results for testing
# Use the real author names that the visualization code expects
authors = ["baum", "thompson", "austen", "dickens", "fitzgerald", "melville", "twain", "wells"]
seeds = [0]  # Just one seed for faster testing
epochs = list(range(1, 51))
datasets = ["train"] + authors

data = []
for author in authors:
    for seed in seeds:
        model_name = f"{author}_model_seed_{seed}"
        for epoch in epochs:
            for dataset in datasets:
                # Generate synthetic loss values
                # Self-loss should be lower than other-loss
                if dataset == "train":
                    loss = 5.0 * np.exp(-epoch / 20) + np.random.normal(0, 0.1)
                elif dataset == author:
                    loss = 4.5 * np.exp(-epoch / 25) + np.random.normal(0, 0.1)
                else:
                    loss = 5.5 * np.exp(-epoch / 15) + np.random.normal(0, 0.1)

                data.append({
                    'model_name': model_name,
                    'train_author': author,
                    'seed': seed,
                    'epochs_completed': epoch,
                    'loss_dataset': dataset,
                    'loss_value': max(loss, 0.5)  # Minimum loss of 0.5
                })

# Create DataFrame
df = pd.DataFrame(data)

# Save as pickle
output_path = data_dir / "test_model_results.pkl"
df.to_pickle(output_path)
print(f"Created test data with {len(df)} rows at {output_path}")

# Also save as CSV for inspection
csv_path = data_dir / "test_model_results.csv"
df.to_csv(csv_path, index=False)
print(f"Also saved as CSV at {csv_path}")

# Create small text samples for testing model training
text_dir = data_dir / "test_texts"
text_dir.mkdir(exist_ok=True)

# Create small sample texts (100 tokens each)
sample_texts = {
    "author1": """The sun rose over the mountains. Birds sang in the trees.
    The day was bright and clear. People walked through the streets.
    Children played in the parks. Life was good in the valley.
    The river flowed gently past. Fish swam in the clear water.
    Flowers bloomed in the gardens. Everything was peaceful here.
    """ * 2,  # Repeat to get more tokens

    "author2": """Dark clouds gathered overhead. Thunder rumbled in distance.
    Rain began to fall heavily. Streets emptied of all people.
    Windows were shuttered tight. The storm approached the city.
    Lightning flashed across sky. Wind howled through the alleys.
    Everyone stayed safely inside. The tempest raged all night.
    """ * 2,  # Repeat to get more tokens
}

for author, text in sample_texts.items():
    file_path = text_dir / f"{author}.txt"
    file_path.write_text(text)
    print(f"Created sample text for {author} at {file_path}")

print("\nTest data creation complete!")