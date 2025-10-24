"""
Tests for model archive download and loading.

These tests verify that:
1. Downloaded model weights can be loaded with transformers
2. Model structure is correct after download
3. Training states can be loaded with PyTorch
4. All model files are present and valid

NO MOCKS - all tests use real model loading, real PyTorch operations.
"""

import pytest
import torch
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config
import subprocess


class TestModelLoading:
    """Test that downloaded models can be loaded and used."""

    @pytest.fixture
    def ensure_baseline_weights(self):
        """Ensure baseline model weights are available."""
        # Check if any baseline model has weights
        models_dir = Path("models")
        baseline_models = list(models_dir.glob("*_tokenizer=gpt2_seed=*"))
        baseline_models = [m for m in baseline_models if "variant=" not in m.name]

        if not baseline_models:
            pytest.skip("No baseline models found")

        # Check if first model has weights
        first_model = baseline_models[0]
        if not (first_model / "model.safetensors").exists():
            pytest.skip("Model weights not downloaded. Run: ./download_model_weights.sh -b")

        return first_model

    def test_load_baseline_model_from_disk(self, ensure_baseline_weights):
        """Verify a baseline model can be loaded with transformers."""
        model_path = ensure_baseline_weights

        # Load model with transformers
        model = GPT2LMHeadModel.from_pretrained(model_path)

        # Verify model structure (custom smaller GPT-2 architecture)
        assert model.config.vocab_size == 50257
        assert model.config.n_layer == 8  # Custom smaller architecture
        assert model.config.n_head == 8
        assert model.config.n_embd == 128

        # Verify model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

        # Expected parameter count for custom architecture: ~7-8M
        assert 5_000_000 < param_count < 10_000_000

    def test_model_forward_pass(self, ensure_baseline_weights):
        """Verify model can run forward pass."""
        model_path = ensure_baseline_weights

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model.eval()

        # Create dummy input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Verify output shape
        assert outputs.logits.shape == (1, 5, 50257)  # (batch, seq_len, vocab_size)

        # Verify outputs are valid (not NaN or inf)
        assert not torch.isnan(outputs.logits).any()
        assert not torch.isinf(outputs.logits).any()

    def test_training_state_loadable(self, ensure_baseline_weights):
        """Verify training_state.pt files are valid PyTorch checkpoints."""
        model_path = ensure_baseline_weights
        state_path = model_path / "training_state.pt"

        # Load training state (weights_only=False required for pickled states)
        state = torch.load(state_path, map_location='cpu', weights_only=False)

        # Verify expected keys
        assert 'optimizer_state_dict' in state
        assert 'epochs_completed' in state
        assert 'random_state' in state or 'np_random_state' in state

        # Verify epoch number is reasonable
        assert 500 <= state['epochs_completed'] <= 10000

    def test_config_files_present(self, ensure_baseline_weights):
        """Verify all required config files are present."""
        model_path = ensure_baseline_weights

        # Check for required files
        assert (model_path / "config.json").exists()
        assert (model_path / "generation_config.json").exists()
        assert (model_path / "loss_logs.csv").exists()
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "training_state.pt").exists()

    def test_multiple_baseline_models_loadable(self):
        """Verify multiple baseline models can be loaded."""
        models_dir = Path("models")
        baseline_models = list(models_dir.glob("*_tokenizer=gpt2_seed=*"))
        baseline_models = [m for m in baseline_models if "variant=" not in m.name]

        if len(baseline_models) < 2:
            pytest.skip("Need at least 2 baseline models")

        # Test first 3 models (or all if less than 3)
        models_to_test = baseline_models[:min(3, len(baseline_models))]

        for model_path in models_to_test:
            # Skip if no weights
            if not (model_path / "model.safetensors").exists():
                continue

            # Load and verify
            model = GPT2LMHeadModel.from_pretrained(model_path)
            assert model.config.vocab_size == 50257

            # Clean up to save memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


class TestVariantModels:
    """Test variant model loading."""

    @pytest.fixture
    def ensure_content_variant(self):
        """Ensure content variant weights are available."""
        models_dir = Path("models")
        content_models = list(models_dir.glob("*_variant=content_tokenizer=gpt2_seed=*"))

        if not content_models:
            pytest.skip("No content variant models found")

        first_model = content_models[0]
        if not (first_model / "model.safetensors").exists():
            pytest.skip("Content variant weights not downloaded. Run: ./download_model_weights.sh -co")

        return first_model

    def test_load_content_variant_model(self, ensure_content_variant):
        """Verify content variant model can be loaded."""
        model_path = ensure_content_variant

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_path)

        # Verify same architecture as baseline
        assert model.config.vocab_size == 50257
        assert model.config.n_layer == 8  # Custom smaller architecture

    def test_all_variants_have_same_architecture(self):
        """Verify all variants use same GPT-2 architecture."""
        models_dir = Path("models")

        # Find one model per variant
        variants = {
            'baseline': list(models_dir.glob("austen_tokenizer=gpt2_seed=0")),
            'content': list(models_dir.glob("austen_variant=content_tokenizer=gpt2_seed=0")),
            'function': list(models_dir.glob("austen_variant=function_tokenizer=gpt2_seed=0")),
            'pos': list(models_dir.glob("austen_variant=pos_tokenizer=gpt2_seed=0"))
        }

        configs = {}
        for variant_name, model_list in variants.items():
            if not model_list:
                continue

            model_path = model_list[0]
            if not (model_path / "model.safetensors").exists():
                continue

            # Load config
            config = GPT2Config.from_pretrained(model_path)
            configs[variant_name] = config

        if len(configs) < 2:
            pytest.skip("Need at least 2 variants to compare")

        # All configs should have same architecture
        first_config = list(configs.values())[0]
        for variant_name, config in configs.items():
            assert config.vocab_size == first_config.vocab_size
            assert config.n_layer == first_config.n_layer
            assert config.n_head == first_config.n_head
            assert config.n_embd == first_config.n_embd


class TestModelCounts:
    """Test that correct number of models are present."""

    def test_baseline_model_count(self):
        """Verify 80 baseline models (8 authors × 10 seeds)."""
        models_dir = Path("models")
        baseline_models = list(models_dir.glob("*_tokenizer=gpt2_seed=*"))
        baseline_models = [m for m in baseline_models if "variant=" not in m.name]

        # Should be 80 total (8 authors × 10 seeds)
        assert len(baseline_models) == 80

    def test_variant_model_counts(self):
        """Verify each variant has 80 models."""
        models_dir = Path("models")

        variants = ['content', 'function', 'pos']
        for variant in variants:
            variant_models = list(models_dir.glob(f"*_variant={variant}_tokenizer=gpt2_seed=*"))

            # Should be 80 per variant
            assert len(variant_models) == 80, f"Expected 80 {variant} models, found {len(variant_models)}"

    def test_total_model_count(self):
        """Verify 320 total models (baseline + 3 variants)."""
        models_dir = Path("models")
        all_models = list(models_dir.glob("*_tokenizer=gpt2_seed=*"))

        # Should be 320 total
        assert len(all_models) == 320

    def test_models_have_weight_files(self):
        """Verify models have both weight files (if downloaded)."""
        models_dir = Path("models")
        all_models = list(models_dir.glob("*_tokenizer=gpt2_seed=*"))

        models_with_weights = 0
        for model_path in all_models:
            safetensors_exists = (model_path / "model.safetensors").exists()
            training_state_exists = (model_path / "training_state.pt").exists()

            # Both files should exist together or both missing
            assert safetensors_exists == training_state_exists, \
                f"Inconsistent weights in {model_path.name}"

            if safetensors_exists:
                models_with_weights += 1

        # If any models have weights, should be all 320
        if models_with_weights > 0:
            assert models_with_weights == 320, \
                f"Expected 0 or 320 models with weights, found {models_with_weights}"


class TestGitTrackedFiles:
    """Test that git-tracked files are present and unchanged."""

    def test_config_files_in_git(self):
        """Verify config files are tracked by git."""
        result = subprocess.run(
            ['git', 'ls-files', 'models/*/config.json'],
            capture_output=True,
            text=True,
            check=True
        )

        config_files = result.stdout.strip().split('\n')
        # Should have 320 config files
        assert len(config_files) == 320

    def test_loss_logs_in_git(self):
        """Verify loss logs are tracked by git."""
        result = subprocess.run(
            ['git', 'ls-files', 'models/*/loss_logs.csv'],
            capture_output=True,
            text=True,
            check=True
        )

        log_files = result.stdout.strip().split('\n')
        # Should have 320 loss log files
        assert len(log_files) == 320

    def test_weight_files_not_in_git(self):
        """Verify weight files are gitignored."""
        result = subprocess.run(
            ['git', 'ls-files', 'models/*/*.safetensors'],
            capture_output=True,
            text=True,
            check=True
        )

        # Should be empty (all gitignored)
        safetensors_files = result.stdout.strip()
        assert safetensors_files == '', "Weight files should be gitignored"

        result = subprocess.run(
            ['git', 'ls-files', 'models/*/training_state.pt'],
            capture_output=True,
            text=True,
            check=True
        )

        training_state_files = result.stdout.strip()
        assert training_state_files == '', "Training state files should be gitignored"

    def test_no_uncommitted_changes_in_models(self):
        """Verify no uncommitted changes in models/ directory."""
        result = subprocess.run(
            ['git', 'status', '--short', 'models/'],
            capture_output=True,
            text=True,
            check=True
        )

        # Should be empty (no changes to tracked files)
        changes = result.stdout.strip()
        assert changes == '', "Git-tracked files in models/ should not be modified"
