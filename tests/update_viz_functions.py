#!/usr/bin/env python
"""
Helper script to systematically update all visualization functions with variant support.
This adds the variant parameter and filtering logic to each function.
"""

import re
from pathlib import Path

# Define the files and their function names
VIZ_FUNCTIONS = {
    'stripplot.py': ['generate_stripplot_figure'],
    't_tests.py': ['generate_t_test_figure', 'generate_t_test_avg_figure'],
    'heatmaps.py': ['generate_loss_heatmap_figure'],
    'mds.py': ['generate_3d_mds_figure'],
    'oz_losses.py': ['generate_oz_losses_figure'],
}

VIZ_DIR = Path('/Users/jmanning/llm-stylometry/llm_stylometry/visualization')

# The variant filtering code to insert after df = pd.read_pickle(data_path)
VARIANT_FILTER_CODE = """
    # Filter by variant
    if variant is None:
        # Baseline: exclude variant models
        if 'variant' in df.columns:
            df = df[df['variant'].isna()].copy()
    else:
        # Specific variant
        if 'variant' not in df.columns:
            raise ValueError(f"No variant column in data")
        df = df[df['variant'] == variant].copy()
"""

# The output path modification code to insert before fig.savefig
OUTPUT_PATH_CODE = """        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path
            output_path = Path(output_path)
            output_path = str(output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}")
"""

def update_function_signature(content, func_name):
    """Add variant=None parameter to function signature."""
    # Find the function definition
    pattern = rf'(def {func_name}\([^)]*?)(\s*)\):'

    def replacement(match):
        params = match.group(1)
        whitespace = match.group(2)
        # Check if variant already added
        if 'variant' in params:
            return match.group(0)
        # Add variant parameter before the closing paren
        return f"{params},{whitespace}\n    variant=None\n):"

    return re.sub(pattern, replacement, content, flags=re.MULTILINE)

def update_docstring(content, func_name):
    """Add variant parameter to docstring."""
    # Find the docstring Args section
    pattern = rf'(def {func_name}.*?""".*?Args:.*?)(Returns:)'

    def replacement(match):
        before_returns = match.group(1)
        returns_section = match.group(2)
        # Check if variant already documented
        if 'variant:' in before_returns:
            return match.group(0)
        # Add variant documentation before Returns
        variant_doc = "        variant: Analysis variant ('content', 'function', 'pos') or None for baseline\n\n    "
        return before_returns + variant_doc + returns_section

    return re.sub(pattern, replacement, content, flags=re.DOTALL)

def add_variant_filtering(content):
    """Add variant filtering code after data loading."""
    # Find df = pd.read_pickle(data_path) and add filtering after it
    pattern = r'(    df = pd\.read_pickle\(data_path\))\n'

    # Check if filtering already added
    if 'Filter by variant' in content:
        return content

    return re.sub(pattern, r'\1\n' + VARIANT_FILTER_CODE + '\n', content)

def add_output_path_modification(content):
    """Add output path modification before fig.savefig."""
    # Find the savefig line
    pattern = r'(    if output_path:\n)(        fig\.savefig)'

    # Check if modification already added
    if 'Add variant suffix' in content:
        return content

    return re.sub(pattern, r'\1' + OUTPUT_PATH_CODE + r'\2', content)

def update_file(filepath, func_names):
    """Update a visualization file with variant support."""
    print(f"\nUpdating {filepath.name}...")

    content = filepath.read_text()
    original_content = content

    # Update each function in the file
    for func_name in func_names:
        print(f"  - {func_name}")
        content = update_function_signature(content, func_name)
        content = update_docstring(content, func_name)

    # Add variant filtering (once per file)
    content = add_variant_filtering(content)

    # Add output path modification (once per file)
    content = add_output_path_modification(content)

    # Write back if changed
    if content != original_content:
        filepath.write_text(content)
        print(f"  ✓ Updated {filepath.name}")
    else:
        print(f"  - No changes needed for {filepath.name}")

def main():
    print("Updating visualization functions with variant support...")
    print("=" * 60)

    for filename, func_names in VIZ_FUNCTIONS.items():
        filepath = VIZ_DIR / filename
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            continue

        try:
            update_file(filepath, func_names)
        except Exception as e:
            print(f"✗ Error updating {filename}: {e}")

    print("\n" + "=" * 60)
    print("✓ All visualization functions updated!")

if __name__ == '__main__':
    main()
