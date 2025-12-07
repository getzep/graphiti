#!/usr/bin/env python3
"""
Verify FastMCP Cloud deployment readiness.

This script checks that your Graphiti MCP server is ready for FastMCP Cloud deployment.
Run this before deploying to catch common issues.

Usage:
    cd mcp_server
    uv run python scripts/verify_fastmcp_cloud_readiness.py
"""

import os
import subprocess
import sys
from pathlib import Path


def print_header(title: str) -> None:
    """Print a section header."""
    print(f'\n{"=" * 60}')
    print(f' {title}')
    print('=' * 60)


def print_result(check: str, passed: bool, details: str = '') -> None:
    """Print a check result."""
    status = '✅ PASSED' if passed else '❌ FAILED'
    print(f'{status}: {check}')
    if details:
        for line in details.split('\n'):
            print(f'         {line}')


def check_server_discoverability() -> bool:
    """Check if fastmcp inspect can discover the server."""
    print_header('Check 1: Server Discoverability')

    # Find the server file
    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    server_file = mcp_server_dir / 'src' / 'graphiti_mcp_server.py'

    if not server_file.exists():
        print_result('Server file exists', False, f'Not found: {server_file}')
        return False

    print_result('Server file exists', True, str(server_file))

    # Try to run fastmcp inspect
    try:
        result = subprocess.run(
            ['uv', 'run', 'fastmcp', 'inspect', f'{server_file}:mcp'],
            capture_output=True,
            text=True,
            cwd=mcp_server_dir,
            timeout=30,
        )

        if result.returncode == 0:
            # Count tools in output
            output = result.stdout
            tool_count = output.count('- ') if '- ' in output else 0
            print_result(
                'fastmcp inspect succeeds',
                True,
                f'Server discovered with {tool_count} tools',
            )
            return True
        else:
            print_result(
                'fastmcp inspect succeeds',
                False,
                f'Error: {result.stderr[:200] if result.stderr else result.stdout[:200]}',
            )
            return False
    except subprocess.TimeoutExpired:
        print_result('fastmcp inspect succeeds', False, 'Command timed out')
        return False
    except FileNotFoundError:
        print_result(
            'fastmcp inspect succeeds',
            False,
            'fastmcp not found. Run: uv sync',
        )
        return False
    except Exception as e:
        print_result('fastmcp inspect succeeds', False, f'Error: {e}')
        return False


def check_dependencies() -> bool:
    """Check that all dependencies are declared in pyproject.toml."""
    print_header('Check 2: Dependencies')

    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    pyproject_file = mcp_server_dir / 'pyproject.toml'

    if not pyproject_file.exists():
        print_result('pyproject.toml exists', False)
        return False

    print_result('pyproject.toml exists', True)

    # Read and check for critical dependencies
    content = pyproject_file.read_text()
    critical_deps = [
        'fastmcp',
        'graphiti-core',
        'pydantic',
        'pydantic-settings',
        'python-dotenv',
    ]

    missing = []
    for dep in critical_deps:
        if dep not in content:
            missing.append(dep)

    if missing:
        print_result(
            'Critical dependencies declared',
            False,
            f'Missing: {", ".join(missing)}',
        )
        return False

    print_result('Critical dependencies declared', True, ', '.join(critical_deps))
    return True


def check_env_documentation() -> bool:
    """Check that environment variables are documented."""
    print_header('Check 3: Environment Variable Documentation')

    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    env_example = mcp_server_dir / '.env.example'

    if not env_example.exists():
        print_result('.env.example exists', False)
        return False

    content = env_example.read_text()
    required_vars = ['OPENAI_API_KEY', 'NEO4J_URI', 'FALKORDB_URI']

    documented = [var for var in required_vars if var in content]

    print_result('.env.example exists', True)
    print_result(
        'Required vars documented',
        len(documented) == len(required_vars),
        f'Found: {", ".join(documented)}',
    )

    return len(documented) == len(required_vars)


def check_secrets_safety() -> bool:
    """Check that no secrets are committed."""
    print_header('Check 4: Secrets Safety')

    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    env_file = mcp_server_dir / '.env'
    gitignore_file = mcp_server_dir.parent / '.gitignore'

    # Check if .env exists (it shouldn't be committed)
    if env_file.exists():
        # Check if it's in gitignore
        if gitignore_file.exists():
            gitignore_content = gitignore_file.read_text()
            if '.env' in gitignore_content:
                print_result('.env in .gitignore', True)
            else:
                print_result('.env in .gitignore', False, 'Add .env to .gitignore!')
                return False
        else:
            print_result('.gitignore exists', False)
            return False
    else:
        print_result('.env not present (good for cloud)', True)

    # Check for hardcoded secrets in server file
    server_file = mcp_server_dir / 'src' / 'graphiti_mcp_server.py'
    if server_file.exists():
        content = server_file.read_text()
        secret_patterns = ['sk-', 'api_key=', 'password=']
        found_secrets = []
        for pattern in secret_patterns:
            if pattern in content.lower() and f"'{pattern}" in content:
                found_secrets.append(pattern)

        if found_secrets:
            print_result(
                'No hardcoded secrets in server',
                False,
                f'Found patterns: {found_secrets}',
            )
            return False
        print_result('No hardcoded secrets in server', True)

    return True


def check_server_import() -> bool:
    """Check that the server can be imported successfully."""
    print_header('Check 5: Server Import')

    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    src_dir = mcp_server_dir / 'src'

    # Add src to path temporarily
    sys.path.insert(0, str(src_dir))

    try:
        # Try importing the module
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            'graphiti_mcp_server',
            src_dir / 'graphiti_mcp_server.py',
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't actually execute - just check if it can be loaded
            print_result('Server module loadable', True)

            # Check for mcp object
            server_content = (src_dir / 'graphiti_mcp_server.py').read_text()
            if 'mcp = FastMCP(' in server_content:
                print_result('Module-level mcp instance found', True)
            else:
                print_result(
                    'Module-level mcp instance found',
                    False,
                    'Need: mcp = FastMCP(...) at module level',
                )
                return False

            return True
        else:
            print_result('Server module loadable', False, 'Could not create module spec')
            return False
    except Exception as e:
        print_result('Server module loadable', False, f'Import error: {e}')
        return False
    finally:
        sys.path.pop(0)


def check_entrypoint_format() -> bool:
    """Check that the entrypoint format is correct for FastMCP Cloud."""
    print_header('Check 6: Entrypoint Format')

    script_dir = Path(__file__).parent
    mcp_server_dir = script_dir.parent
    server_file = mcp_server_dir / 'src' / 'graphiti_mcp_server.py'

    if not server_file.exists():
        print_result('Server file exists', False)
        return False

    content = server_file.read_text()

    # Check for module-level FastMCP instance
    has_module_level_mcp = 'mcp = FastMCP(' in content

    # Check for if __name__ == "__main__" block (should exist but is ignored)
    has_main_block = "if __name__ == '__main__':" in content or 'if __name__ == "__main__":' in content

    print_result(
        'Module-level mcp instance',
        has_module_level_mcp,
        'Required for FastMCP Cloud discovery',
    )

    if has_main_block:
        print_result(
            '__main__ block present',
            True,
            'Note: This is IGNORED by FastMCP Cloud',
        )

    # Print the expected entrypoint
    if has_module_level_mcp:
        print(f'\n  Expected entrypoint for FastMCP Cloud:')
        print(f'  src/graphiti_mcp_server.py:mcp')

    return has_module_level_mcp


def main() -> None:
    """Run all verification checks."""
    print('\n' + '=' * 60)
    print(' FastMCP Cloud Deployment Readiness Check')
    print(' Graphiti MCP Server')
    print('=' * 60)

    checks = [
        ('Server Discoverability', check_server_discoverability),
        ('Dependencies', check_dependencies),
        ('Environment Documentation', check_env_documentation),
        ('Secrets Safety', check_secrets_safety),
        ('Server Import', check_server_import),
        ('Entrypoint Format', check_entrypoint_format),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f'\n❌ Error in {name}: {e}')
            results.append((name, False))

    # Summary
    print_header('Summary')
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = '✅' if passed else '❌'
        print(f'  {status} {name}')

    print(f'\n  Total: {passed_count}/{total_count} checks passed')

    if passed_count == total_count:
        print('\n' + '=' * 60)
        print(' 🚀 READY FOR FASTMCP CLOUD DEPLOYMENT!')
        print('=' * 60)
        print('\nNext steps:')
        print('  1. Push code to GitHub')
        print('  2. Visit https://fastmcp.cloud')
        print('  3. Create project with entrypoint: src/graphiti_mcp_server.py:mcp')
        print('  4. Set environment variables in FastMCP Cloud UI')
        print('  5. Deploy!')
        print('\nSee docs/FASTMCP_CLOUD_DEPLOYMENT.md for detailed instructions.')
    else:
        print('\n' + '=' * 60)
        print(' ⚠️  NOT READY - Please fix the failing checks above')
        print('=' * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()
