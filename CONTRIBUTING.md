# Contributing to LLM Stylometry

Thank you for your interest in contributing to LLM Stylometry! This project applies language model stylometry to analyze authorship patterns in literary texts. We welcome contributions that improve the codebase, add new features, fix bugs, or enhance documentation.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Guidelines](#development-guidelines)
- [Testing Philosophy](#testing-philosophy)
- [Reporting Bugs](#reporting-bugs)
- [Communication](#communication)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

Unacceptable behavior should be reported to [contextualdynamics@gmail.com](mailto:contextualdynamics@gmail.com).

## How to Contribute

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/llm-stylometry.git
   cd llm-stylometry
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following our development guidelines
5. **Test your changes** thoroughly
6. **Commit your changes** with descriptive messages
7. **Push to your fork** and submit a pull request

### What to Contribute

We welcome contributions in several areas:

- **Bug fixes**: Identify and fix issues in existing code
- **New features**: Add functionality that aligns with the project's goals
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve README, docstrings, or add examples
- **Tests**: Expand test coverage or improve test quality
- **Code refactoring**: Improve code organization and readability

## Development Guidelines

### Code Style

- **Formatting**: Use `black` for code formatting:
  ```bash
  black .
  ```
- **Naming conventions**:
  - Use descriptive variable and function names
  - Follow existing naming patterns in the codebase
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`

### Code Organization

- Follow the existing package structure:
  ```
  llm_stylometry/
  ├── core/           # Experiment configuration and training
  ├── data/           # Data loading and preprocessing
  ├── models/         # Model initialization
  ├── analysis/       # Statistical analysis
  └── visualization/  # Figure generation
  ```
- Keep functions focused and single-purpose
- Add docstrings to all public functions and classes
- Use type hints where appropriate

### Dependencies

- Minimize new dependencies
- Only add dependencies that are:
  - Well-maintained
  - Widely used
  - Necessary for the feature
- Update `requirements.txt` when adding dependencies
- Document why the dependency is needed

## Testing Philosophy

### Core Principles

**Use real models, data, and outputs—no mocks.**

Our testing philosophy prioritizes real-world validation over unit test isolation. This approach ensures:
- External APIs work correctly (e.g., Anthropic, OpenAI, Hugging Face)
- Models can be downloaded and used properly
- Responses are in expected formats
- Database operations succeed
- Files are created and read correctly
- Figures render properly

### Writing Tests

When writing tests, follow these guidelines:

1. **Use real data and models**:
   ```python
   # Good: Use actual small model
   model = GPT2LMHeadModel.from_pretrained('gpt2')

   # Bad: Mock the model
   model = MagicMock()
   ```

2. **Keep tests fast**:
   - Use small datasets (synthetic test data)
   - Use tiny models (e.g., GPT-2 with minimal layers)
   - Target < 5 minutes for entire test suite

3. **Test actual outputs**:
   ```python
   # Good: Verify real file was created
   fig = generate_figure(data, 'output.pdf')
   assert Path('output.pdf').exists()
   assert Path('output.pdf').stat().st_size > 1000

   # Bad: Mock file creation
   with patch('pathlib.Path.exists', return_value=True):
       ...
   ```

4. **Test edge cases**:
   - Empty inputs
   - Missing files
   - Invalid parameters
   - Boundary conditions

5. **Run tests locally before submitting**:
   ```bash
   pytest tests/
   ```

### Test Coverage

Ensure new features have comprehensive test coverage:
- Happy path (expected usage)
- Error handling
- Edge cases
- Cross-platform compatibility (if applicable)

## Reporting Bugs

When reporting a bug, please include:

1. **Short summary**: Brief description of the issue
2. **Reproduction steps**: Minimal code snippet to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - Operating system
   - Python version
   - Package versions (from `pip list`)

### Bug Report Template

```markdown
## Bug Description
[Brief description]

## Steps to Reproduce
```python
# Minimal code example
```

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.0]
- PyTorch: [e.g., 2.0.0]
```

## Communication

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Pull Requests**: Use PRs for code contributions
- **Email**: For sensitive issues or questions, contact [contextualdynamics@gmail.com](mailto:contextualdynamics@gmail.com)

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Follow the PR template checklist
4. Write clear commit messages
5. Reference related issues (e.g., "Fixes #123")
6. Be responsive to review feedback

### Review Process

- PRs require approval from a maintainer
- Reviewers will check:
  - Code quality and style
  - Test coverage
  - Documentation completeness
  - Adherence to project guidelines
- Address feedback promptly and professionally

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Email the maintainers
- Check existing documentation in the README

Thank you for contributing to LLM Stylometry! 🎉
