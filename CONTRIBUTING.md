# Contributing to Graphiti

We're thrilled you're interested in contributing to Graphiti! As firm believers in the power of open source collaboration, we're committed to building not just a tool, but a vibrant community where developers of all experience levels can make meaningful contributions.

When I first joined this project, I was overwhelmed trying to figure out where to start. Someone eventually pointed me to a random "good first issue," but I later discovered there were multiple ways I could have contributed that would have better matched my skills and interests.

We've restructured our contribution paths to solve this problem:

# Four Ways to Get Involved
- Pick up existing issues: Our developers regularly tag issues with "help wanted" and "good first issue." These are pre-vetted tasks with clear scope and someone ready to help you if you get stuck.

- Create your own tickets: See something that needs fixing? Have an idea for an improvement? You don't need permission to identify problems. Create a ticket describing what you'd like to change and why it matters.

- Share your use cases: Sometimes the most valuable contribution isn't code. If you're using our project in an interesting way, add it to the examples folder. This helps others discover new possibilities and counts as a meaningful contribution. We regularly feature compelling examples in our blog posts and videos - your work might be showcased to the broader community!

- Help others in Discord: Join our Discord community and pitch in at the helpdesk. Answering questions and helping troubleshoot issues is an incredibly valuable contribution that benefits everyone. The knowledge you share today saves someone hours of frustration tomorrow.

# What happens next?
Regardless of which path you choose, the process works the same way: fork the repo, make your changes in a branch, and submit a PR. We've included more detailed technical instructions below.

Your perspective is valuable, no matter how you choose to contribute. Come say hello in our [Discord server](https://discord.gg/2JbGZQZT)  - whether you're ready to contribute or just want to learn more, we're happy to have you!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/getzep/graphiti
   cd graphiti
   ```
3. Set up your development environment:
   - Ensure you have Python 3.10+ installed.
   - Install Poetry: https://python-poetry.org/docs/#installation
   - Install project dependencies:
     ```
     make install
     ```
   - To run integration tests, set the appropriate environment variables
     ```
     export TEST_OPENAI_API_KEY=...
     export TEST_OPENAI_MODEL=...

     export NEO4J_URI=neo4j://...
     export NEO4J_USER=...
     export NEO4J_PASSWORD=...
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
