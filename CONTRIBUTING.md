# Contributing to Graphiti

We're excited you're interested in contributing to Graphiti! This document outlines the process for contributing to our project. We welcome contributions from everyone, whether you're fixing a typo, improving documentation, or adding a new feature.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/getzep/graphiti
   cd graphiti
   ```
3. Set up your development environment:
   - Ensure you have Python 3.8+ installed.
   - Install Poetry: https://python-poetry.org/docs/#installation
   - Install project dependencies:
     ```
     make install
     ```

## Making Changes

1. Create a new branch for your changes:
   ```
   git checkout -b your-branch-name
   ```
2. Make your changes in the codebase.
3. Write or update tests as necessary.
4. Run the tests to ensure they pass:
   ```
   make test
   ```
5. Format your code:
   ```
   make format
   ```
6. Run linting checks:
   ```
   make lint
   ```

## Submitting Changes

1. Commit your changes:
   ```
   git commit -m "Your detailed commit message"
   ```
2. Push to your fork:
   ```
   git push origin your-branch-name
   ```
3. Submit a pull request through the GitHub website to https://github.com/getzep/graphiti.

## Pull Request Guidelines

- Provide a clear title and description of your changes.
- Include any relevant issue numbers in the PR description.
- Ensure all tests pass and there are no linting errors.
- Update documentation if you're changing functionality.

## Code Style and Quality

We use several tools to maintain code quality:

- Ruff for linting and formatting
- Mypy for static type checking
- Pytest for testing

Before submitting a pull request, please run:

```
make check
```

This command will format your code, run linting checks, and execute tests.

## Reporting Bugs

Use the GitHub issue tracker at https://github.com/getzep/graphiti/issues to report bugs. When filing an issue, please include:

- A clear title and description
- As much relevant information as possible
- A code sample or an executable test case demonstrating the expected behavior that is not occurring

## Feature Requests

Feature requests are welcome. Please provide a clear description of the feature and why it would be beneficial to the project. You can submit feature requests through the GitHub issue tracker.

## Questions?

If you have any questions, feel free to open an issue or reach out to the maintainers through the GitHub repository.

Thank you for contributing to Graphiti!
