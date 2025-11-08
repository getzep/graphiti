# Task Completion Checklist

When you complete a coding task, follow these steps to ensure quality:

## 1. Format Code
```bash
make format
```
This will:
- Sort imports using ruff (isort rules)
- Format code to 100-character line length
- Apply single-quote style
- Format code in docstrings

## 2. Lint Code
```bash
make lint
```
This will:
- Run ruff checks for code quality issues
- Run pyright type checking on `graphiti_core/`
- Identify any style violations or type errors

Fix any issues reported by the linter.

## 3. Run Tests
```bash
# Run unit tests (default)
make test

# OR run all tests including integration tests
uv run pytest
```

Ensure all tests pass. If you:
- Modified existing functionality: Verify related tests still pass
- Added new functionality: Consider adding new tests
- Fixed a bug: Consider adding a regression test

## 4. Integration Testing (if applicable)
If your changes affect:
- Database interactions
- LLM integrations
- External service calls

Run integration tests:
```bash
# Ensure environment variables are set
export TEST_OPENAI_API_KEY=...
export TEST_URI=neo4j://...
export TEST_USER=...
export TEST_PASSWORD=...

# Run integration tests
uv run pytest -k "_int"
```

## 5. Type Checking
Pyright should have passed during `make lint`, but if you added new code, verify:
- All function parameters have type hints
- Return types are specified
- No `Any` types unless necessary
- Pydantic models are properly defined

## 6. Run Complete Check
Run the comprehensive check command:
```bash
make check
```
This runs format, lint, and test in sequence. All should pass.

## 7. Documentation Updates (if needed)
Consider if your changes require:
- README.md updates (for user-facing features)
- CLAUDE.md updates (for development patterns)
- Docstring additions/updates
- Example code in `examples/` folder
- Comments for complex logic

## 8. Git Commit
Only commit if all checks pass:
```bash
git add <files>
git commit -m "Descriptive commit message"
```

## 9. PR Preparation (if submitting changes)
Before creating a PR:
- Ensure `make check` passes completely
- Review your changes for any debug code or comments
- Check for any TODO items you added
- Verify no sensitive data (API keys, passwords) in code
- Consider if changes need an issue/RFC (>500 LOC changes require discussion)

## Quick Reference
Most common workflow:
```bash
# After making changes
make check

# If all passes, commit
git add .
git commit -m "Your message"
```

## Special Cases

### Server Changes
If you modified `server/` code:
```bash
cd server/
make format
make lint
make test
```

### MCP Server Changes
If you modified `mcp_server/` code:
```bash
cd mcp_server/
# Test with Docker
docker-compose up
```

### Large Architectural Changes
- Create GitHub issue (RFC) first
- Discuss technical design and justification
- Get feedback before implementing >500 LOC changes
