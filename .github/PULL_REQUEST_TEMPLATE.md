# Pull Request

## Description
<!-- Provide a brief description of the changes in this PR -->

## Related Issue
<!-- If this PR addresses an issue, link it here (e.g., "Fixes #123") -->

## Type of Change
<!-- Mark the appropriate option with an "x" -->

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement
- [ ] Test addition or modification

## Checklist

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Code has been formatted with `black` (run `black .` from project root)
- [ ] Variable and function names are descriptive and follow existing conventions
- [ ] Code is well-organized and follows the package structure

### Testing
- [ ] **No mock objects or functions are used in tests** (tests use real models, data, and outputs)
- [ ] All new functionality has comprehensive test coverage
- [ ] Tests use small datasets/models and complete quickly (< 5 minutes total)
- [ ] Tests verify actual outputs (files created, correct formats, expected content)
- [ ] All existing tests pass locally (`pytest tests/`)
- [ ] Tests work across platforms (Linux, macOS, Windows if applicable)

### Documentation
- [ ] Updated relevant documentation (README.md, docstrings, etc.)
- [ ] Added/updated code comments where necessary
- [ ] Examples are provided for new features

### Dependencies
- [ ] No unnecessary dependencies added
- [ ] If new dependencies added, updated `requirements.txt`
- [ ] Verified compatibility with existing dependencies

### Git Hygiene
- [ ] Commits have descriptive messages
- [ ] No sensitive information (passwords, keys, tokens) in code or commits
- [ ] `.gitignore` updated if new file types should be excluded
- [ ] No large files committed (model weights, large datasets, etc.)

### Functional Requirements
- [ ] Changes have been tested manually
- [ ] Scripts run without errors
- [ ] Generated outputs (figures, models, etc.) are correct
- [ ] No breaking changes to existing functionality (or documented if necessary)

## Additional Notes
<!-- Any additional information reviewers should know -->
