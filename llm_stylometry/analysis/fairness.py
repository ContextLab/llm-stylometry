"""Fairness-based loss thresholding for variant model comparisons."""

import pandas as pd
import numpy as np


def compute_fairness_threshold(df, min_epochs=500):
    """
    Compute fairness threshold for variant models.

    For each model, finds minimum training loss within min_epochs.
    Returns the maximum of all these minimums as the fairness threshold.

    This ensures all models are compared at the same training loss level,
    preventing unfair comparisons where some models converged to higher
    losses than others.

    Args:
        df: DataFrame with model results (must have 'loss_dataset', 'epochs_completed',
            'loss_value', 'train_author', 'seed' columns)
        min_epochs: Minimum number of epochs to consider (default: 500)

    Returns:
        float: Fairness threshold (maximum of all models' minimum losses)

    Raises:
        ValueError: If insufficient data or missing required columns

    Examples:
        >>> df = pd.read_pickle('data/model_results_function.pkl')
        >>> threshold = compute_fairness_threshold(df, min_epochs=500)
        >>> print(f"Fairness threshold: {threshold:.4f}")
        Fairness threshold: 1.2720
    """
    # Validate required columns
    required_cols = ['loss_dataset', 'epochs_completed', 'loss_value', 'train_author', 'seed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter for training losses only
    train_df = df[df['loss_dataset'] == 'train'].copy()

    if len(train_df) == 0:
        raise ValueError("No training loss data found (loss_dataset == 'train')")

    # Filter for epochs <= min_epochs
    train_df = train_df[train_df['epochs_completed'] <= min_epochs]

    if len(train_df) == 0:
        raise ValueError(f"No data found with epochs_completed <= {min_epochs}")

    # Group by (train_author, seed) - each unique model
    grouped = train_df.groupby(['train_author', 'seed'])

    if len(grouped) == 0:
        raise ValueError("No models found after grouping by train_author and seed")

    # For each model, find minimum loss_value
    min_losses = grouped['loss_value'].min()

    if len(min_losses) == 0 or min_losses.isna().all():
        raise ValueError("Could not compute minimum losses (all NaN)")

    # Return maximum of all minimums
    threshold = min_losses.max()

    if np.isnan(threshold):
        raise ValueError("Fairness threshold is NaN")

    return float(threshold)


def apply_fairness_threshold(df, threshold, use_first_crossing=True):
    """
    Truncate model data at fairness threshold.

    For each model, finds first (or last) epoch where training loss <= threshold
    and keeps all data UP TO AND INCLUDING that epoch. This ensures all models
    are compared at the same training loss level (the fairness threshold), even
    though they reach it at different epochs.

    Args:
        df: DataFrame with model results
        threshold: Fairness threshold computed by compute_fairness_threshold()
        use_first_crossing: If True, truncate at first epoch crossing threshold.
                          If False, truncate at last epoch <= threshold (default: True)

    Returns:
        DataFrame: Truncated data with all models stopped when they first reach the fairness threshold

    Raises:
        ValueError: If models cannot reach threshold or missing data

    Examples:
        >>> df = pd.read_pickle('data/model_results_function.pkl')
        >>> threshold = compute_fairness_threshold(df)
        >>> df_fair = apply_fairness_threshold(df, threshold)
        >>> # Verify each model's final training loss is at the threshold
        >>> for (author, seed), group in df_fair.groupby(['train_author', 'seed']):
        ...     train_data = group[group['loss_dataset'] == 'train']
        ...     final_loss = train_data['loss_value'].iloc[-1]
        ...     assert final_loss <= threshold + 0.001
    """
    # Validate required columns
    required_cols = ['loss_dataset', 'epochs_completed', 'loss_value', 'train_author', 'seed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create copy to avoid modifying original
    df = df.copy()

    # Find training loss trajectories and cutoff epochs for each model
    cutoff_epochs = {}

    for (author, seed), group in df.groupby(['train_author', 'seed']):
        # Get training losses for this model
        train_data = group[group['loss_dataset'] == 'train'].sort_values('epochs_completed')

        if len(train_data) == 0:
            raise ValueError(f"No training data for model {author} seed {seed}")

        # Find epochs where loss <= threshold
        below_threshold = train_data[train_data['loss_value'] <= threshold]

        if len(below_threshold) == 0:
            # Model never reaches threshold - use all epochs
            cutoff_epoch = train_data['epochs_completed'].max()
        else:
            if use_first_crossing:
                # Use first epoch crossing threshold
                cutoff_epoch = below_threshold['epochs_completed'].min()
            else:
                # Use last epoch <= threshold
                cutoff_epoch = below_threshold['epochs_completed'].max()

        cutoff_epochs[(author, seed)] = cutoff_epoch

    # Truncate all rows (train + eval datasets) at cutoff epoch for each model
    truncated_rows = []

    for (author, seed), group in df.groupby(['train_author', 'seed']):
        cutoff = cutoff_epochs.get((author, seed))
        if cutoff is not None:
            # Keep only rows where epochs_completed <= cutoff
            truncated = group[group['epochs_completed'] <= cutoff]
            truncated_rows.append(truncated)

    if len(truncated_rows) == 0:
        raise ValueError("No data remaining after applying fairness threshold")

    # Combine all truncated data
    result = pd.concat(truncated_rows, ignore_index=True)

    return result
