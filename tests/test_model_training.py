#!/usr/bin/env python
"""Test model training with tiny models and datasets."""

import pytest
import sys
import tempfile
from pathlib import Path
import torch
import pandas as pd
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelTraining:
    """Test model training functionality with tiny models."""

    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_texts_dir = Path(__file__).parent / "data" / "test_texts"

        # Check for GPU availability (not required but good to know)
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Testing on device: {cls.device}")

    def test_tiny_model_creation(self):
        """Test creating a tiny GPT-2 model for testing."""
        # Create tiny config
        config = GPT2Config(
            vocab_size=1000,  # Tiny vocabulary
            n_positions=128,  # Short sequences
            n_embd=32,       # Tiny embeddings
            n_layer=2,       # Only 2 layers
            n_head=2,        # Only 2 attention heads
            n_ctx=128
        )

        # Create model
        model = GPT2LMHeadModel(config)

        # Verify model size
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count < 500000, f"Model too large: {param_count} parameters"

        # Save model config for reference
        config_path = Path(self.temp_dir) / "tiny_model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        assert config_path.exists(), "Config not saved"

    def test_tokenization(self):
        """Test tokenization of sample texts."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Read test texts
        test_files = list(self.test_texts_dir.glob("*.txt"))
        assert len(test_files) > 0, "No test text files found"

        for text_file in test_files:
            text = text_file.read_text()
            tokens = tokenizer.encode(text, truncation=True, max_length=128)

            assert len(tokens) > 0, f"No tokens generated for {text_file.name}"
            assert len(tokens) <= 128, f"Tokens not truncated for {text_file.name}"

    def test_quick_training_step(self):
        """Test a single training step with tiny model."""
        # Create tiny model
        config = GPT2Config(
            vocab_size=1000,
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=1
        )
        model = GPT2LMHeadModel(config)
        model.to(self.device)

        # Create random input
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(self.device)

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        assert loss is not None, "No loss computed"
        assert loss.item() > 0, "Loss should be positive"

        # Backward pass
        loss.backward()

        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "No gradients computed"

    def test_loss_computation(self):
        """Test loss computation for different texts."""
        # Create tiny model
        config = GPT2Config(
            vocab_size=50257,  # Use real vocab size for tokenizer compatibility
            n_positions=128,
            n_embd=32,
            n_layer=2,
            n_head=2
        )
        model = GPT2LMHeadModel(config)
        model.eval()  # Set to eval mode
        model.to(self.device)

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Test with different texts
        texts = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Hello world, this is a test."
        ]

        losses = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                losses.append(outputs.loss.item())

        # Verify losses are computed
        assert all(loss > 0 for loss in losses), "All losses should be positive"
        assert len(set(losses)) == len(losses), "Losses should be different for different texts"

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Create model
        config = GPT2Config(
            vocab_size=1000,
            n_positions=64,
            n_embd=16,
            n_layer=1,
            n_head=1
        )
        model = GPT2LMHeadModel(config)

        # Save model
        model_path = Path(self.temp_dir) / "test_model"
        model.save_pretrained(model_path)

        # Load model
        loaded_model = GPT2LMHeadModel.from_pretrained(model_path)

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2, f"Parameter names don't match: {n1} vs {n2}"
            assert torch.allclose(p1, p2), f"Parameter values don't match for {n1}"

    def test_generate_synthetic_results(self):
        """Test generation of synthetic model results DataFrame."""
        # Generate synthetic training results
        authors = ["test_author1", "test_author2"]
        epochs = range(1, 11)
        seeds = [0]

        data = []
        for author in authors:
            for seed in seeds:
                for epoch in epochs:
                    for eval_author in authors:
                        loss = 5.0 / epoch if author == eval_author else 6.0 / epoch
                        data.append({
                            'model_name': f'{author}_seed{seed}',
                            'train_author': author,
                            'seed': seed,
                            'epochs_completed': epoch,
                            'loss_dataset': eval_author,
                            'loss_value': loss
                        })

        df = pd.DataFrame(data)

        # Verify DataFrame structure
        assert len(df) > 0, "No data generated"
        assert 'loss_value' in df.columns, "Missing loss_value column"
        assert df['loss_value'].min() > 0, "Loss values should be positive"

        # Save for inspection
        df_path = Path(self.temp_dir) / "synthetic_results.pkl"
        df.to_pickle(df_path)
        assert df_path.exists(), "DataFrame not saved"

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        import shutil
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])