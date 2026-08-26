# Contributing to Graphiti

Welcome, and thank you for your interest in Graphiti! Whether you have found a bug, have an idea
you would like to see built, spotted a confusing doc, or simply have a question, we are glad you
are here. This guide will help you find the right starting point so we can respond quickly and
put your contribution to good use.

## Where to start

The [issue chooser](https://github.com/getzep/graphiti/issues/new/choose) will point you to the
right form:

- **Bug:** Something behaves differently than the docs or your reasonable expectations suggest.
- **Feature:** You would like new functionality, or an improvement to how something works today.
- **Documentation:** Something is incorrect, unclear, or missing from our docs or examples.
- **Question:** You would like help understanding or using Graphiti. Share the versions you are
  on and what you have already tried, and we can get to a useful answer sooner.
- **Security vulnerability:** Please do **not** open a public issue. Report it privately using
  the steps in [SECURITY.md](SECURITY.md) so we can fix it before it is widely known.

Not sure which one fits? Pick your best guess and file it — a maintainer will happily re-route it.
We would much rather hear from you than have you wonder whether it was worth reporting.

Looking for other ways to help? All of these are genuinely valuable:

- Pick up an issue tagged [`help wanted`](https://github.com/getzep/graphiti/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)
  or [`good first issue`](https://github.com/getzep/graphiti/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).
  These are pre-vetted and scoped, and someone is around to help if you get stuck.
- Share how you use Graphiti by adding to [`examples/`](https://github.com/getzep/graphiti/tree/main/examples).
  Good examples help more people than you might expect.
- Answer questions and help troubleshoot in GitHub Issues. The knowledge you share today saves
  someone hours tomorrow.

### Reporting a bug

The fastest fixes start with a report someone else can reproduce. The more of the following you
can share, the sooner we can help:

- A minimal, self-contained code sample or test case
- What you expected to happen, and what actually happened
- The complete error or traceback, if there was one
- Your Graphiti and Python versions, operating system, and how you installed
- Which part is affected: core library, MCP server, REST server, or documentation
- Your database backend and version
- Your LLM, embedding, or reranking provider and model, when they are involved

If you cannot fill in everything, file what you have — we will ask about anything else we need.

One request: please scrub API keys, credentials, and private data before posting. Once it is in a
public issue, it is public.

### Proposing a feature

Tell us the story of what you are trying to accomplish: what are you building, what is getting in
your way, and what would make your life easier? Starting with the problem rather than a specific
implementation gives us room to find the best solution together.

Small improvements can move ahead once a maintainer has weighed in on the issue. Larger features
deserve a design conversation first, so you do not invest a weekend in an approach we would have
to redesign in review. We treat a feature as large when it involves any of:

- A new database driver
- A new LLM, embedding, or reranking provider
- A new API endpoint or public capability
- A major architectural or data-model change
- A change likely to exceed 500 lines

Your feature issue is the design discussion — there is no separate RFC to file. Fill in the
proposal, alternatives, and impact sections, and a maintainer will add `rfc-approved` once the
design is settled. Until then the issue or pull request may carry `needs-rfc`, which simply means
the conversation is still open.

Please hold off on a large implementation until the design is approved. Prototypes are a great way
to explore an idea, so feel free to build one and share what you learned — just expect that we may
ask it to stay in draft while the design comes together.

### What we prioritize

Bug fixes to existing functionality get the most attention and the fastest review. For anything
substantial, sharing your approach on the issue first is time well spent: it keeps two people from
building the same thing and helps your work fit Graphiti's architecture from the start.

## Labels and who does what

Our labels are meant to make the state of your issue obvious at a glance:

| Category | Labels | What it tells you |
| --- | --- | --- |
| Type | `bug`, `feature`, `question`, `documentation` | What kind of issue this is |
| Process | `intake/needs-info`, `needs-rfc`, `rfc-approved`, `needs-tests`, `needs-rework` | What needs to happen next |
| Area | `area/core`, `area/mcp`, `area/server`, `area/docs` | Which part of Graphiti is affected |

A process label is never a judgment about you or your work — it is a note about the next step. If a
label appears and you are not sure what it is asking for, just say so on the issue and we will
explain.

You may still see `enhancement` and `slop-detected` on older items. We now use `feature` and the
more actionable `needs-rework` instead. To be clear: using AI assistance is fine and does not by
itself mean a contribution needs rework.

Here is who does what:

| Role | What they do |
| --- | --- |
| You, the contributor | Pick a form, share context, talk through substantial changes, and link pull requests to issues |
| Intake automation | Sorts and routes new issues, asks for missing details, and flags policy gaps — it never approves designs or closes your contribution |
| Maintainers | Approve designs, set priority, mark `good first issue` and `help wanted`, review code, and make the call on merging |

## Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/getzep/graphiti
   cd graphiti
   ```
3. Set up your development environment:

   - Ensure you have Python 3.10+ installed.
   - Install uv: https://docs.astral.sh/uv/getting-started/installation/
   - Install project dependencies:
     ```
     make install
     ```
   - To run integration tests, set the appropriate environment variables

     ```
     export TEST_OPENAI_API_KEY=...
     export TEST_OPENAI_MODEL=...
     export TEST_ANTHROPIC_API_KEY=...

     # For Neo4j
     export TEST_URI=neo4j://...
     export TEST_USER=...
     export TEST_PASSWORD=...
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

A few things that help us review your work quickly:

- Give it a clear title, and explain both the problem and your solution.
- Link the bug or feature with `Fixes #<issue-number>`. Docs-only and routine maintenance changes
  can skip this when the reason speaks for itself.
- For a large feature, link to a feature issue that already has `rfc-approved`.
- Add or update tests for behavior changes. If tests do not make sense here, just tell us why.
- Run `make check`, and mention anything you were not able to run — that is useful to know, not
  something to hide.
- Update the docs when behavior or public interfaces change.
- Keep credentials, API keys, and customer data out of commits.
- Sign the Contributor License Agreement when the bot prompts you.

## Code Style and Quality

We use several tools to maintain code quality:

- Ruff for linting and formatting
- Pyright for static type checking
- Pytest for testing

Before submitting a pull request, please run:

```
make check
```

This command will format your code, run linting checks, and execute tests.

## Third-Party Integrations

When contributing integrations for third-party services (LLM providers, embedding services, databases, etc.), please follow these patterns:

### Optional Dependencies

All third-party integrations must be optional dependencies to keep the core library lightweight. Follow this pattern:

1. **Add to `pyproject.toml`**: Define your dependency as an optional extra AND include it in the dev extra:
   ```toml
   [project.optional-dependencies]
   your-service = ["your-package>=1.0.0"]
   dev = [
       # ... existing dev dependencies
       "your-package>=1.0.0",  # Include all optional extras here
       # ... other dependencies
   ]
   ```

2. **Use TYPE_CHECKING pattern**: In your integration module, import dependencies conditionally:
   ```python
   from typing import TYPE_CHECKING
   
   if TYPE_CHECKING:
       import your_package
       from your_package import SomeType
   else:
       try:
           import your_package
           from your_package import SomeType
       except ImportError:
           raise ImportError(
               'your-package is required for YourServiceClient. '
               'Install it with: pip install graphiti-core[your-service]'
           ) from None
   ```

3. **Benefits of this pattern**:
   - Fast startup times (no import overhead during type checking)
   - Clear error messages with installation instructions
   - Proper type hints for development
   - Consistent user experience

4. **Do NOT**:
   - Add optional imports to `__init__.py` files
   - Use direct imports without error handling
   - Include optional dependencies in the main `dependencies` list

### Integration Structure

- Place LLM clients in `graphiti_core/llm_client/`
- Place embedding clients in `graphiti_core/embedder/`
- Place database drivers in `graphiti_core/driver/`
- Follow existing naming conventions (e.g., `your_service_client.py`)

### Adding a Graph Driver

Graphiti's driver layer is backend-agnostic. To add support for a new graph database, mirror the existing drivers in
`graphiti_core/driver/` and keep the implementation split between the top-level driver and provider-specific
operations.

1. Add the new provider to `graphiti_core/driver/driver.py` in `GraphProvider`.
2. Create `graphiti_core/driver/<backend>_driver.py` implementing the `GraphDriver` interface:
   `execute_query()`, `session()`, `close()`, `build_indices_and_constraints()`, and `delete_all_indexes()`.
3. Add `graphiti_core/driver/<backend>/operations/` and implement the operations interfaces from
   `graphiti_core/driver/operations/`:
   `EntityNodeOperations`, `EpisodeNodeOperations`, `CommunityNodeOperations`, `SagaNodeOperations`,
   `EntityEdgeOperations`, `EpisodicEdgeOperations`, `CommunityEdgeOperations`, `HasEpisodeEdgeOperations`,
   `NextEpisodeEdgeOperations`, `SearchOperations`, and `GraphMaintenanceOperations`.
4. Expose those concrete operations from the driver via the corresponding `@property` accessors on `GraphDriver`.
5. Add provider-specific query variants to `graphiti_core/models/nodes/node_db_queries.py` and
   `graphiti_core/models/edges/edge_db_queries.py`.
6. If the backend needs connection or transaction management, implement a matching `GraphDriverSession`.
7. Register the backend dependency in `pyproject.toml` under `[project.optional-dependencies]` and add tests under
   `tests/driver/`.

For reference implementations, start with `graphiti_core/driver/neo4j_driver.py`,
`graphiti_core/driver/falkordb_driver.py`, and `graphiti_core/driver/neptune_driver.py`
(`graphiti_core/driver/kuzu_driver.py` is deprecated — don't model new drivers on it).

### Testing

- Add comprehensive tests in the appropriate `tests/` subdirectory
- Mark integration tests with `_int` suffix if they require external services
- Include both unit tests and integration tests where applicable

# Questions?

Stuck on a contribution or have a half-formed idea? Open a [GitHub issue](https://github.com/getzep/graphiti/issues) and say hello. Whether you're ready to contribute or just want to learn more, we're happy to have you! You'll find both maintainers and fellow contributors ready to help.

Thank you for contributing to Graphiti!
