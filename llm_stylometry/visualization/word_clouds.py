"""Generate word cloud visualizations using wordcloud library."""

import pickle
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from wordcloud import WordCloud

from llm_stylometry.core.constants import AUTHORS


def extract_average_weights(
    results_df,
    feature_names,
    author: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract and average feature weights across all CV splits.

    Args:
        results_df: Results DataFrame with classifier objects
        feature_names: List of feature names
        author: Specific author or None for overall

    Returns:
        Dictionary mapping word → weight
    """
    # Get unique classifiers (one per split)
    unique_splits = results_df['split_id'].unique()

    # Extract weights from each split's classifier
    all_weights = []

    for split_id in unique_splits:
        # Get classifier from this split (all rows in same split have same classifier)
        split_rows = results_df[results_df['split_id'] == split_id]
        if len(split_rows) == 0:
            continue

        clf = split_rows.iloc[0]['classifier']

        # Extract feature weights
        try:
            weights_dict = clf.get_feature_weights(feature_names)

            if author is None:
                # Overall weights
                split_weights = weights_dict['overall']
            else:
                # Author-specific weights
                split_weights = weights_dict[author]

            all_weights.append(split_weights)
        except Exception as e:
            print(f"Warning: Could not extract weights from split {split_id}: {e}")
            continue

    if not all_weights:
        raise ValueError("No weights could be extracted from classifiers")

    # Average weights across all splits
    averaged_weights = {}
    for word in feature_names:
        word_weights = [w[word] for w in all_weights if word in w]
        if word_weights:
            averaged_weights[word] = np.mean(word_weights)

    return averaged_weights


def generate_word_cloud_figure(
    data_path: str,
    author: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    font: str = 'Helvetica',
    variant: Optional[str] = None,
    max_words: int = 100
):
    """
    Generate word cloud figure using wordcloud library.

    Args:
        data_path: Path to classifier results pkl
        author: Specific author (e.g., "baum") or None for overall
        output_path: Path to save PDF (optional)
        figsize: Figure size
        font: Font family
        variant: Analysis variant or None for baseline
        max_words: Maximum words to display

    Returns:
        matplotlib figure object

    Examples:
        >>> # Overall word cloud
        >>> fig = generate_word_cloud_figure(
        ...     data_path='data/classifier_results/baseline.pkl',
        ...     author=None
        ... )
        >>> # Author-specific word cloud
        >>> fig = generate_word_cloud_figure(
        ...     data_path='data/classifier_results/baseline.pkl',
        ...     author='baum'
        ... )
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load results
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results_df = data['results']
    feature_names = data['feature_names']

    # Extract averaged weights
    weights = extract_average_weights(results_df, feature_names, author)

    # Use absolute values for word cloud (magnitude matters)
    abs_weights = {word: abs(weight) for word, weight in weights.items()}

    # Define color based on author
    if author is None:
        color = 'black'
    else:
        # Use same color palette as all_losses figure
        author_idx = AUTHORS.index(author.lower())
        base_colors = sns.color_palette("tab10", n_colors=len(AUTHORS))
        color = base_colors[author_idx]

    # Create color function for wordcloud
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if isinstance(color, str):
            return color
        else:
            # Convert RGB tuple to hex
            return mcolors.rgb2hex(color)

    # Initialize WordCloud
    wc = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=max_words,
        relative_scaling=0.5,
        color_func=color_func,
        prefer_horizontal=0.7
    )

    # Generate from frequencies
    wc.generate_from_frequencies(abs_weights)

    # Extract layout and render as vectorized text using matplotlib
    layout = wc.layout_

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, wc.width)
    ax.set_ylim(0, wc.height)
    ax.axis('off')

    # Render words as matplotlib text objects (vectorized)
    for (word, count), font_size, (x, y), orientation, wc_color in layout:
        # Use our color function
        text_color = color_func(word, font_size, (x, y), orientation)

        ax.text(
            x, wc.height - y,  # Flip y-axis to match image coordinates
            word,
            fontsize=font_size * 0.5,  # Scale down font size for better fit
            color=text_color,
            rotation=orientation,
            ha='center',
            va='center',
            family=font
        )

    plt.tight_layout(pad=0)

    # Save if output path provided
    if output_path is None:
        if author is None:
            if variant is None:
                output_path = "paper/figs/source/wordcloud_overall_baseline.pdf"
            else:
                output_path = f"paper/figs/source/wordcloud_overall_{variant}.pdf"
        else:
            if variant is None:
                output_path = f"paper/figs/source/wordcloud_{author}_baseline.pdf"
            else:
                output_path = f"paper/figs/source/wordcloud_{author}_{variant}.pdf"

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format='pdf', bbox_inches='tight')

    return fig
