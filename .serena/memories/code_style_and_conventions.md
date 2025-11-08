# Code Style and Conventions

## Formatting and Linting

### Ruff Configuration
- **Tool**: Ruff (handles both linting and formatting)
- **Line length**: 100 characters
- **Quote style**: Single quotes (`'`)
- **Indentation**: Spaces (not tabs)
- **Docstring code format**: Enabled (formats code in docstrings)

### Linting Rules (Ruff)
Enabled rule sets:
- `E` - pycodestyle errors
- `F` - Pyflakes
- `UP` - pyupgrade (Python version upgrades)
- `B` - flake8-bugbear (common bugs)
- `SIM` - flake8-simplify (simplification suggestions)
- `I` - isort (import sorting)

Ignored rules:
- `E501` - Line too long (handled by line-length setting)

### Type Checking
- **Tool**: Pyright
- **Python version**: 3.10+
- **Type checking mode**: `basic` for main project, `standard` for server
- **Scope**: Main type checking focuses on `graphiti_core/` directory
- **Type hints**: Required and enforced

## Code Conventions

### General Guidelines
- Use type hints for all function parameters and return values
- Follow PEP 8 style guide (enforced by Ruff)
- Use Pydantic models for data validation and structure
- Prefer async/await for I/O operations
- Use descriptive variable and function names

### Python Version
- Minimum supported: Python 3.10
- Maximum supported: Python 3.x (< 4.0)

### Import Organization
Imports are automatically organized by Ruff using isort rules:
1. Standard library imports
2. Third-party imports
3. Local application imports

### Documentation
- Use docstrings for classes and public methods
- Keep README.md and CLAUDE.md up to date
- Add examples to `examples/` folder for new features
- Document breaking changes and migrations

### Testing Conventions
- Use pytest for all tests
- Async tests use `pytest-asyncio`
- Integration tests must have `_int` suffix in filename or test name
- Unit tests should not require external services
- Use fixtures from `conftest.py`
- Parallel execution supported via `pytest-xdist`

### Naming Conventions
- Classes: PascalCase (e.g., `Graphiti`, `AddEpisodeResults`)
- Functions/methods: snake_case (e.g., `add_episode`, `build_indices`)
- Constants: UPPER_SNAKE_CASE
- Private methods/attributes: prefix with underscore (e.g., `_internal_method`)

### Error Handling
- Use custom exceptions from `graphiti_core/errors.py`
- Provide meaningful error messages
- Use `tenacity` for retry logic on external service calls

### LLM Provider Support
- The codebase supports multiple LLM providers
- Best compatibility with services supporting structured output (OpenAI, Gemini)
- Smaller models may cause schema validation issues
- Always validate LLM outputs against expected schemas

## Configuration and Dependencies
- Use `pyproject.toml` for all project configuration
- Pin minimum versions in dependencies
- Optional features go in `[project.optional-dependencies]`
- Development dependencies go in `dev` extra
