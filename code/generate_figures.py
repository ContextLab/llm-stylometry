#!/usr/bin/env python
"""
Comprehensive CLI for LLM Stylometry: model training and figure generation.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import safe_print for Windows compatibility
from llm_stylometry.cli_utils import safe_print, format_header, is_windows


def train_models(max_gpus=None, no_confirm=False, resume=False, variant=None):
    """Train all models from scratch or resume from checkpoints."""
    safe_print("\n" + "=" * 60)
    if resume:
        safe_print("Resuming Model Training from Checkpoints")
    else:
        safe_print("Training Models from Scratch")
    safe_print("=" * 60)
    warning = "[WARNING]" if is_windows() else "⚠️"
    # Check device availability
    import torch
    device_info = ""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device_info = f"CUDA GPUs available: {gpu_count}"
    elif torch.backends.mps.is_available():
        device_info = "Apple Metal Performance Shaders (MPS) available"
    else:
        device_info = "CPU only (training will be slow)"

    safe_print(f"\n{warning}  Warning: This will train 80 models (8 authors × 10 seeds)")
    safe_print(f"   Device: {device_info}")
    if variant:
        safe_print(f"   Variant: {variant}_only")
    else:
        safe_print("   Variant: baseline")
    safe_print("   Training time depends on hardware (hours on GPU, days on CPU)")

    if not no_confirm:
        response = input("\nProceed with training? [y/N]: ")
        if response.lower() != 'y':
            safe_print("Training cancelled.")
            return False
    else:
        safe_print("\nSkipping confirmation (--no-confirm flag set)")
        safe_print("Starting training...")

    # Handle models directory based on resume flag
    import shutil
    models_dir = Path('models')

    if not resume:
        # Remove existing models directory to train from scratch
        if models_dir.exists():
            safe_print("\nRemoving existing models directory...")
            shutil.rmtree(models_dir)
            safe_print("Existing models removed.")

        # Also remove existing model results file
        model_results_path = Path('data/model_results.pkl')
        if model_results_path.exists():
            safe_print("Removing existing model_results.pkl...")
            model_results_path.unlink()
    else:
        # When resuming, keep existing models and check their status
        if models_dir.exists():
            safe_print("\nResuming from existing models directory...")
        else:
            safe_print("\nNo existing models found. Starting fresh training...")
            resume = False  # Fall back to fresh training if no models exist

    # Prepare data if needed
    if not Path('data/cleaned').exists():
        safe_print("\nCleaning data first...")
        result = subprocess.run([sys.executable, 'code/clean.py'], capture_output=True)
        if result.returncode != 0:
            safe_print(f"Error cleaning data: {result.stderr.decode()}")
            return False

    # Train models
    safe_print("\nTraining models...")
    try:
        # Set environment variables for training
        env = os.environ.copy()
        env['DISABLE_TQDM'] = '1'  # Disable progress bars in subprocess
        # Only disable multiprocessing if we have a single GPU or non-GPU device
        # With multiple GPUs, we want parallel training
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count <= 1:
                env['NO_MULTIPROCESSING'] = '1'
                safe_print("Single GPU detected - using sequential mode")
            else:
                safe_print(f"Multiple GPUs detected ({gpu_count}) - using parallel training")
        else:
            # Non-CUDA device (CPU or MPS)
            env['NO_MULTIPROCESSING'] = '1'
            safe_print("Non-CUDA device - using sequential mode")
        # Set PyTorch memory management for better GPU memory usage
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Pass through max GPUs limit if specified
        if max_gpus:
            env['MAX_GPUS'] = str(max_gpus)
            safe_print(f"Limiting to {max_gpus} GPU(s)")
        # Pass through resume flag if specified
        if resume:
            env['RESUME_TRAINING'] = '1'
        # Pass through analysis variant if specified
        if variant:
            env['ANALYSIS_VARIANT'] = variant
            safe_print(f"Training {variant} variant models")
        # Run without capturing output so we can see progress
        result = subprocess.run([sys.executable, 'code/main.py'], env=env, check=False)
        if result.returncode != 0:
            safe_print(f"Error: Training script exited with code {result.returncode}")
            return False
    except Exception as e:
        safe_print(f"Error running training script: {e}")
        return False

    # Consolidate results
    safe_print("\nConsolidating model results...")
    try:
        result = subprocess.run([sys.executable, 'code/consolidate_model_results.py'], check=False)
        if result.returncode != 0:
            safe_print(f"Error: Consolidation script exited with code {result.returncode}")
            return False
    except Exception as e:
        safe_print(f"Error running consolidation script: {e}")
        return False

    checkmark = "[OK]" if is_windows() else "✓"
    safe_print(f"\n{checkmark} Model training complete!")
    return True


def generate_figure(figure_name, data_path='data/model_results.pkl', output_dir='paper/figs/source'):
    """Generate a specific figure."""
    from llm_stylometry.visualization import (
        generate_all_losses_figure,
        generate_stripplot_figure,
        generate_t_test_figure,
        generate_t_test_avg_figure,
        generate_loss_heatmap_figure,
        generate_3d_mds_figure,
        generate_oz_losses_figure
    )

    figure_map = {
        '1a': ('all_losses', generate_all_losses_figure, 'all_losses.pdf'),
        '1b': ('stripplot', generate_stripplot_figure, 'stripplot.pdf'),
        '2a': ('t_test', generate_t_test_figure, 't_test.pdf'),
        '2b': ('t_test_avg', generate_t_test_avg_figure, 't_test_avg.pdf'),
        '3': ('heatmap', generate_loss_heatmap_figure, 'average_loss_heatmap.pdf'),
        '4': ('mds', generate_3d_mds_figure, '3d_MDS_plot.pdf'),
        '5': ('oz', generate_oz_losses_figure, 'oz_losses.pdf'),
    }

    if figure_name not in figure_map:
        safe_print(f"Unknown figure: {figure_name}")
        safe_print(f"Available figures: {', '.join(figure_map.keys())}")
        return False

    name, func, filename = figure_map[figure_name]
    output_path = Path(output_dir) / filename

    safe_print(f"Generating Figure {figure_name.upper()}: {name}...")
    try:
        kwargs = {'data_path': data_path, 'output_path': str(output_path)}
        if name in ['all_losses', 'stripplot', 't_test', 't_test_avg', 'oz']:
            kwargs['show_legend'] = False
        fig = func(**kwargs)
        plt.close(fig)
        checkmark = "[OK]" if is_windows() else "✓"
        safe_print(f"  {checkmark} Generated: {output_path}")
        return True
    except Exception as e:
        cross = "[FAIL]" if is_windows() else "✗"
        safe_print(f"  {cross} Error: {str(e)}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='LLM Stylometry CLI: Train models and generate figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Generate all figures from pre-computed results
  %(prog)s --figure 1a        # Generate only Figure 1A
  %(prog)s --figure 4         # Generate only Figure 4 (MDS plot)
  %(prog)s --train            # Train models from scratch, then generate figures
  %(prog)s --list             # List available figures
        """
    )

    parser.add_argument(
        '--figure', '-f',
        help='Generate specific figure (1a, 1b, 2a, 2b, 3, 4, 5)',
        default=None
    )

    parser.add_argument(
        '--train', '-t',
        action='store_true',
        help='Train models from scratch before generating figures'
    )

    parser.add_argument(
        '--data', '-d',
        help='Path to model_results.pkl (default: data/model_results.pkl)',
        default='data/model_results.pkl'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output directory for figures (default: paper/figs/source)',
        default='paper/figs/source'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available figures'
    )

    parser.add_argument(
        '--max-gpus', '-g',
        type=int,
        help='Maximum number of GPUs to use for training (default: all available)',
        default=None
    )

    parser.add_argument(
        '--no-confirm', '-y',
        action='store_true',
        help='Skip confirmation prompts (useful for non-interactive mode)'
    )

    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from existing checkpoints (use with --train)'
    )

    parser.add_argument(
        '--variant',
        choices=['content', 'function', 'pos'],
        default=None,
        help='Analysis variant for training (content-only, function-only, or POS-only)'
    )

    args = parser.parse_args()

    if args.list:
        safe_print("\nAvailable figures:")
        safe_print("  1a - Figure 1A: Training curves (all_losses.pdf)")
        safe_print("  1b - Figure 1B: Strip plot (stripplot.pdf)")
        safe_print("  2a - Figure 2A: Individual t-tests (t_test.pdf)")
        safe_print("  2b - Figure 2B: Average t-test (t_test_avg.pdf)")
        safe_print("  3  - Figure 3: Confusion matrix heatmap (average_loss_heatmap.pdf)")
        safe_print("  4  - Figure 4: 3D MDS plot (3d_MDS_plot.pdf)")
        safe_print("  5  - Figure 5: Oz authorship analysis (oz_losses.pdf)")
        return 0

    safe_print(format_header("LLM Stylometry CLI", 60))

    # Validate --resume flag usage
    if args.resume and not args.train:
        safe_print("\nWarning: --resume flag is ignored without --train flag")
        args.resume = False

    # Train models if requested
    if args.train:
        if not train_models(max_gpus=args.max_gpus, no_confirm=args.no_confirm, resume=args.resume, variant=args.variant):
            return 1
        # Update data path to use newly generated results
        args.data = 'data/model_results.pkl'

    # Check for data file
    data_file = Path(args.data)
    if not data_file.exists():
        safe_print(f"\nERROR: Required data file not found: {args.data}")
        safe_print("Please ensure you have the consolidated model results.")
        safe_print("You can train models from scratch using: --train")
        return 1

    checkmark = "[OK]" if is_windows() else "✓"
    safe_print(f"\n{checkmark} Found model results data: {args.data}")

    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate specific figure if requested
    if args.figure:
        success = generate_figure(args.figure, args.data, args.output)
        return 0 if success else 1

    safe_print("\n" + "=" * 60)
    safe_print("Generating Figures")
    safe_print("=" * 60)

    # Import visualization functions
    from llm_stylometry.visualization import (
        generate_all_losses_figure,
        generate_stripplot_figure,
        generate_t_test_figure,
        generate_t_test_avg_figure,
        generate_loss_heatmap_figure,
        generate_3d_mds_figure,
        generate_oz_losses_figure
    )

    figures = [
        ('Figure 1A: Training curves',
         lambda: generate_all_losses_figure(
             data_path=args.data,
             output_path=f'{args.output}/all_losses.pdf',
             show_legend=False
         )),
        ('Figure 1B: Strip plot',
         lambda: generate_stripplot_figure(
             data_path=args.data,
             output_path=f'{args.output}/stripplot.pdf',
             show_legend=False
         )),
        ('Figure 2A: Individual t-tests',
         lambda: generate_t_test_figure(
             data_path=args.data,
             output_path=f'{args.output}/t_test.pdf',
             show_legend=False
         )),
        ('Figure 2B: Average t-test',
         lambda: generate_t_test_avg_figure(
             data_path=args.data,
             output_path=f'{args.output}/t_test_avg.pdf',
             show_legend=False
         )),
        ('Figure 3: Confusion matrix',
         lambda: generate_loss_heatmap_figure(
             data_path=args.data,
             output_path=f'{args.output}/average_loss_heatmap.pdf'
         )),
        ('Figure 4: 3D MDS plot',
         lambda: generate_3d_mds_figure(
             data_path=args.data,
             output_path=f'{args.output}/3d_MDS_plot.pdf'
         )),
        ('Figure 5: Oz losses',
         lambda: generate_oz_losses_figure(
             data_path=args.data,
             output_path=f'{args.output}/oz_losses.pdf',
             show_legend=False
         )),
    ]

    success_count = 0
    failed_figures = []

    for description, generate_func in figures:
        safe_print(f"\nGenerating {description}...")
        try:
            fig = generate_func()
            plt.close(fig)
            checkmark = "[OK]" if is_windows() else "✓"
            safe_print(f"  {checkmark} Generated successfully")
            success_count += 1
        except Exception as e:
            cross = "[FAIL]" if is_windows() else "✗"
            safe_print(f"  {cross} Error: {str(e)[:100]}")
            failed_figures.append(description)

    # Verify outputs
    safe_print("\n" + "=" * 60)
    safe_print("Verifying Output Files")
    safe_print("=" * 60)

    expected_files = [
        (f'{args.output}/all_losses.pdf', 'Figure 1A'),
        (f'{args.output}/stripplot.pdf', 'Figure 1B'),
        (f'{args.output}/t_test.pdf', 'Figure 2A'),
        (f'{args.output}/t_test_avg.pdf', 'Figure 2B'),
        (f'{args.output}/average_loss_heatmap.pdf', 'Figure 3'),
        (f'{args.output}/3d_MDS_plot.pdf', 'Figure 4'),
        (f'{args.output}/oz_losses.pdf', 'Figure 5'),
    ]

    for file_path, name in expected_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            checkmark = "[OK]" if is_windows() else "✓"
            safe_print(f"  {checkmark} {name}: {size_kb:.1f} KB")
        else:
            cross = "[FAIL]" if is_windows() else "✗"
            safe_print(f"  {cross} {name}: NOT FOUND")

    # Summary
    safe_print("\n" + "=" * 60)
    if success_count == len(figures):
        checkmark = "[OK]" if is_windows() else "✓"
        safe_print(f"{checkmark} All figures generated successfully!")
        safe_print("=" * 60)
        safe_print(f"\nFigures are saved in: {args.output}/")
    else:
        warning = "[WARNING]" if is_windows() else "⚠"
        safe_print(f"{warning} Generated {success_count}/{len(figures)} figures")
        if failed_figures:
            safe_print("\nFailed figures:")
            for fig in failed_figures:
                safe_print(f"  - {fig}")
        safe_print("\nPlease check the error messages above.")

    return 0 if success_count == len(figures) else 1


if __name__ == "__main__":
    sys.exit(main())