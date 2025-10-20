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
    import glob
    models_dir = Path('models')

    if not resume:
        # Remove only the models for the current variant
        if models_dir.exists():
            if variant:
                # For variant training, only remove models with that variant
                pattern = f"*_variant={variant}_*"
                variant_models = list(models_dir.glob(pattern))
                if variant_models:
                    safe_print(f"\nRemoving existing {variant} variant models ({len(variant_models)} directories)...")
                    for model_path in variant_models:
                        shutil.rmtree(model_path)
                    safe_print(f"Existing {variant} variant models removed.")
                else:
                    safe_print(f"\nNo existing {variant} variant models found.")
            else:
                # For baseline training, only remove baseline models (no variant in name)
                baseline_models = [p for p in models_dir.iterdir()
                                   if p.is_dir() and '_variant=' not in p.name]
                if baseline_models:
                    safe_print(f"\nRemoving existing baseline models ({len(baseline_models)} directories)...")
                    for model_path in baseline_models:
                        shutil.rmtree(model_path)
                    safe_print("Existing baseline models removed.")
                else:
                    safe_print("\nNo existing baseline models found.")
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
        # Build consolidation command with variant if specified
        consolidate_cmd = [sys.executable, 'code/consolidate_model_results.py']
        if variant:
            consolidate_cmd.extend(['--variant', variant])

        result = subprocess.run(consolidate_cmd, check=False)
        if result.returncode != 0:
            safe_print(f"Error: Consolidation script exited with code {result.returncode}")
            return False
    except Exception as e:
        safe_print(f"Error running consolidation script: {e}")
        return False

    checkmark = "[OK]" if is_windows() else "✓"
    safe_print(f"\n{checkmark} Model training complete!")
    return True


def generate_figure(figure_name, data_path='data/model_results.pkl', output_dir='paper/figs/source', variant=None):
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

    # Skip Figure 5 for variants with clear message
    if figure_name == '5' and variant is not None:
        safe_print(f"Skipping Figure 5 (Oz losses) for {variant} variant - requires contested/non-Oz datasets")
        return True  # Return True to indicate intentional skip, not failure

    name, func, filename = figure_map[figure_name]
    output_path = Path(output_dir) / filename

    safe_print(f"Generating Figure {figure_name.upper()}: {name}...")
    try:
        kwargs = {'data_path': data_path, 'output_path': str(output_path), 'variant': variant}
        if name in ['all_losses', 'stripplot', 't_test', 't_test_avg', 'oz']:
            kwargs['show_legend'] = False
        fig = func(**kwargs)

        # Handle None return (intentional skip)
        if fig is None:
            checkmark = "[OK]" if is_windows() else "✓"
            safe_print(f"  {checkmark} Skipped (not applicable for this variant)")
            return True

        plt.close(fig)
        checkmark = "[OK]" if is_windows() else "✓"
        safe_print(f"  {checkmark} Generated: {output_path}")
        return True
    except Exception as e:
        cross = "[FAIL]" if is_windows() else "✗"
        safe_print(f"  {cross} Error: {str(e)}")
        return False


def run_single_classification_variant(args_tuple):
    """
    Run classification for a single variant (module-level for multiprocessing).

    Args:
        args_tuple: (variant, output_dir, skip_experiment) tuple

    Returns:
        (variant_name, success, error_message) tuple
    """
    variant, output_dir, skip_experiment = args_tuple
    variant_name = variant if variant else "baseline"

    try:
        from llm_stylometry.classification import run_classification_experiment
        from llm_stylometry.core.constants import AUTHORS
        from llm_stylometry.visualization import (
            generate_classification_accuracy_figure,
            generate_word_cloud_figure
        )
        from pathlib import Path

        # Determine result path
        result_path = f"data/classifier_results/{variant_name}.pkl"

        # Run experiment only if results don't exist or skip_experiment is False
        if not skip_experiment or not Path(result_path).exists():
            result_path = run_classification_experiment(
                variant=variant,
                max_splits=1000,
                seed=42
            )

        # Generate word clouds (per-variant)
        # Overall word cloud
        wc_overall = f"{output_dir}/wordcloud_overall_{variant_name}.pdf"
        generate_word_cloud_figure(
            data_path=result_path,
            author=None,
            output_path=wc_overall,
            variant=variant
        )

        # Per-author word clouds
        for author in AUTHORS:
            wc_author = f"{output_dir}/wordcloud_{author}_{variant_name}.pdf"
            generate_word_cloud_figure(
                data_path=result_path,
                author=author,
                output_path=wc_author,
                variant=variant
            )

        return (variant_name, True, None)

    except Exception as e:
        import traceback
        return (variant_name, False, traceback.format_exc())


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
        help='Path to model_results.pkl (auto-detected based on variant if not specified)',
        default=None
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

    parser.add_argument(
        '--no-fairness',
        action='store_true',
        help='Disable fairness-based loss thresholding for variant figures (default: fairness enabled for variants)'
    )

    parser.add_argument(
        '--classify',
        action='store_true',
        help='Run text classification experiment instead of GPT-2 training'
    )

    parser.add_argument(
        '--classify-variant',
        action='append',
        dest='classify_variants',
        help='Variant(s) for classification (can specify multiple for parallel execution)'
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
        safe_print("  5  - Figure 5: Oz authorship analysis (oz_losses.pdf) [baseline only]")
        return 0

    safe_print(format_header("LLM Stylometry CLI", 60))

    # Validate --resume flag usage
    if args.resume and not args.train:
        safe_print("\nWarning: --resume flag is ignored without --train flag")
        args.resume = False

    # Auto-detect data path if not specified
    if args.data is None:
        if args.variant:
            args.data = f'data/model_results_{args.variant}.pkl'
        else:
            args.data = 'data/model_results.pkl'

    # Train models if requested
    if args.train:
        if not train_models(max_gpus=args.max_gpus, no_confirm=args.no_confirm, resume=args.resume, variant=args.variant):
            return 1

    # Run classification experiment if requested
    if args.classify:
        safe_print("\n" + "=" * 60)
        safe_print("Running Text Classification Experiment")
        safe_print("=" * 60)

        from llm_stylometry.classification import run_classification_experiment
        from llm_stylometry.core.constants import AUTHORS
        from multiprocessing import Pool, cpu_count

        # Determine which variants to run
        variants_to_run = []
        if args.classify_variants:
            # Variants explicitly specified via --classify-variant flags
            for v in args.classify_variants:
                if v == 'baseline':
                    variants_to_run.append(None)
                else:
                    variants_to_run.append(v)
        else:
            # No --classify-variant flags: use default behavior
            if args.variant:
                # Single variant from --variant flag (e.g., from -co)
                variants_to_run = [args.variant]
            else:
                # No variants specified at all: default to baseline only
                variants_to_run = [None]

        safe_print(f"\nVariants to run: {[v if v else 'baseline' for v in variants_to_run]}")
        safe_print(f"Max CV splits per variant: 1,000")

        if len(variants_to_run) > 1:
            safe_print(f"Running {len(variants_to_run)} variants in parallel on {cpu_count()} CPUs")

        # Determine if we should skip experiment (load existing results)
        # Skip only if --train flag is NOT set
        skip_experiment = not args.train

        # Prepare arguments for parallel execution
        variant_args = [(v, args.output, skip_experiment) for v in variants_to_run]

        # Run experiments (parallel if multiple variants)
        try:
            if len(variants_to_run) == 1:
                # Single variant - run directly
                variant_name, success, error = run_single_classification_variant(variant_args[0])
                if not success:
                    safe_print(f"\n✗ ERROR: Classification failed for {variant_name}")
                    safe_print(error)
                    return 1
                else:
                    safe_print(f"\n✓ Classification complete for {variant_name}")
            else:
                # Multiple variants - run in parallel
                with Pool(processes=min(len(variants_to_run), cpu_count())) as pool:
                    results = pool.map(run_single_classification_variant, variant_args)

                # Check results
                failed = []
                for variant_name, success, error in results:
                    if not success:
                        failed.append(variant_name)
                        safe_print(f"\n✗ ERROR: Classification failed for {variant_name}")
                        safe_print(error)
                    else:
                        safe_print(f"\n✓ Classification complete for {variant_name}")

                if failed:
                    safe_print(f"\n✗ {len(failed)}/{len(variants_to_run)} classifications failed")
                    return 1

            # Generate single grouped accuracy bar chart combining all conditions
            safe_print("\nGenerating grouped accuracy bar chart...")
            from llm_stylometry.visualization import generate_classification_accuracy_figure
            acc_output = f"{args.output}/classification_accuracy.pdf"
            generate_classification_accuracy_figure(output_path=acc_output)
            safe_print(f"✓ Generated: {acc_output}")

            safe_print("\n" + "=" * 60)
            safe_print("✓ All classification experiments complete!")
            safe_print("=" * 60)
            return 0

        except Exception as e:
            safe_print(f"\n✗ ERROR: Classification experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

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
        success = generate_figure(args.figure, args.data, args.output, variant=args.variant)
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
             show_legend=False,
             variant=args.variant,
             apply_fairness=not args.no_fairness
         )),
        ('Figure 1B: Strip plot',
         lambda: generate_stripplot_figure(
             data_path=args.data,
             output_path=f'{args.output}/stripplot.pdf',
             show_legend=False,
             variant=args.variant,
             apply_fairness=not args.no_fairness
         )),
        ('Figure 2A: Individual t-tests',
         lambda: generate_t_test_figure(
             data_path=args.data,
             output_path=f'{args.output}/t_test.pdf',
             show_legend=False,
             variant=args.variant
         )),
        ('Figure 2B: Average t-test',
         lambda: generate_t_test_avg_figure(
             data_path=args.data,
             output_path=f'{args.output}/t_test_avg.pdf',
             show_legend=False,
             variant=args.variant
         )),
        ('Figure 3: Confusion matrix',
         lambda: generate_loss_heatmap_figure(
             data_path=args.data,
             output_path=f'{args.output}/average_loss_heatmap.pdf',
             variant=args.variant,
             apply_fairness=not args.no_fairness
         )),
        ('Figure 4: 3D MDS plot',
         lambda: generate_3d_mds_figure(
             data_path=args.data,
             output_path=f'{args.output}/3d_MDS_plot.pdf',
             variant=args.variant,
             apply_fairness=not args.no_fairness
         )),
    ]

    # Only include Figure 5 for baseline (no variant)
    if args.variant is None:
        figures.append(
            ('Figure 5: Oz losses',
             lambda: generate_oz_losses_figure(
                 data_path=args.data,
                 output_path=f'{args.output}/oz_losses.pdf',
                 show_legend=False,
                 variant=args.variant,
                 apply_fairness=not args.no_fairness
             ))
        )
    else:
        safe_print(f"\nNote: Skipping Figure 5 for {args.variant} variant (requires contested/non-Oz datasets)")

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
    ]

    # Only verify Figure 5 for baseline
    if args.variant is None:
        expected_files.append((f'{args.output}/oz_losses.pdf', 'Figure 5'))

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