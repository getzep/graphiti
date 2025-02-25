# Contributing to Graphiti

We're thrilled you're interested in contributing to Graphiti! As firm believers in the power of open source collaboration, we're committed to building not just a tool, but a vibrant community where developers of all experience levels can make meaningful contributions.

When I first joined this project, I was overwhelmed trying to figure out where to start. Someone eventually pointed me to a random "good first issue," but I later discovered there were multiple ways I could have contributed that would have better matched my skills and interests.

We've restructured our contribution paths to solve this problem:

# Four Ways to Get Involved

### Pick Up Existing Issues
Our developers regularly tag issues with "help wanted" and "good first issue." These are pre-vetted tasks with clear scope and someone ready to help you if you get stuck.

### Create Your Own Tickets
See something that needs fixing? Have an idea for an improvement? You don't need permission to identify problems. The people closest to the pain are often best positioned to describe the solution.

For **feature requests**, tell us the story of what you're trying to accomplish. What are you working on? What's getting in your way? What would make your life easier? Submit these through our [GitHub issue tracker](https://github.com/getzep/graphiti/issues) with a "Feature Request" label.

For **bug reports**, we need enough context to reproduce the problem. Use the [GitHub issue tracker](https://github.com/getzep/graphiti/issues) and include:
- A clear title that summarizes the specific problem
- What you were trying to do when you encountered the bug
- What you expected to happen
- What actually happened
- A code sample or test case that demonstrates the issue

### Share Your Use Cases
Sometimes the most valuable contribution isn't code. If you're using our project in an interesting way, add it to the [examples](https://github.com/getzep/graphiti/tree/main/examples) folder. This helps others discover new possibilities and counts as a meaningful contribution. We regularly feature compelling examples in our blog posts and videos - your work might be showcased to the broader community!

### Help Others in Discord
Join our [Discord server](https://discord.gg/2JbGZQZT)  community and pitch in at the helpdesk. Answering questions and helping troubleshoot issues is an incredibly valuable contribution that benefits everyone. The knowledge you share today saves someone hours of frustration tomorrow.

## What happens next?
Once you've found an issue tagged with "good first issue" or "help wanted," or prepared an example to share, here's how to turn that into a contribution:

1. Share your approach in the issue discussion or [Discord](https://discord.gg/2JbGZQZT) before diving deep into code. This helps ensure your solution adheres to the architecture of Graphiti from the start and saves you from potential rework. 

2. Fork the repo, make your changes in a branch, and submit a PR. We've included more detailed technical instructions below; be open to feedback during review.

## Setup
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

# Questions?
Stuck on a contribution or have a half-formed idea? Come say hello in our [Discord server](https://discord.gg/2JbGZQZT). Whether you're ready to contribute or just want to learn more, we're happy to have you! It's faster than GitHub issues and you'll find both maintainers and fellow contributors ready to help.

Thank you for contributing to Graphiti!
