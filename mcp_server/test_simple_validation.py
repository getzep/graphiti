#!/usr/bin/env python3
"""
Simple validation test for the refactored Graphiti MCP Server.
Tests basic functionality quickly without timeouts.
"""

import subprocess
import sys
import time


def test_server_startup():
    """Test that the refactored server starts up successfully."""
    print('🚀 Testing Graphiti MCP Server Startup...')

    try:
        # Start the server and capture output
        process = subprocess.Popen(
            ['uv', 'run', 'graphiti_mcp_server.py', '--transport', 'stdio'],
            env={
                'NEO4J_URI': 'bolt://localhost:7687',
                'NEO4J_USER': 'neo4j',
                'NEO4J_PASSWORD': 'demodemo',
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for startup logs
        startup_output = ''
        for _ in range(50):  # Wait up to 5 seconds
            if process.poll() is not None:
                break
            time.sleep(0.1)

            # Check if we have output
            try:
                line = process.stderr.readline()
                if line:
                    startup_output += line
                    print(f'   📋 {line.strip()}')

                    # Check for success indicators
                    if 'Graphiti client initialized successfully' in line:
                        print('   ✅ Graphiti service initialization: SUCCESS')
                        success = True
                        break

            except Exception:
                continue
        else:
            print('   ⚠️  Timeout waiting for initialization')
            success = False

        # Clean shutdown
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        return success

    except Exception as e:
        print(f'   ❌ Server startup failed: {e}')
        return False


def test_import_validation():
    """Test that all refactored modules import correctly."""
    print('\n🔍 Testing Module Import Validation...')

    modules_to_test = [
        'config_manager',
        'llm_config',
        'embedder_config',
        'neo4j_config',
        'server_config',
        'graphiti_service',
        'queue_service',
        'entity_types',
        'response_types',
        'formatting',
        'utils',
    ]

    success_count = 0

    for module in modules_to_test:
        try:
            result = subprocess.run(
                ['python', '-c', f"import {module}; print('✅ {module}')"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                print(f'   ✅ {module}: Import successful')
                success_count += 1
            else:
                print(f'   ❌ {module}: Import failed - {result.stderr.strip()}')

        except subprocess.TimeoutExpired:
            print(f'   ❌ {module}: Import timeout')
        except Exception as e:
            print(f'   ❌ {module}: Import error - {e}')

    print(f'   📊 Import Results: {success_count}/{len(modules_to_test)} modules successful')
    return success_count == len(modules_to_test)


def test_syntax_validation():
    """Test that all Python files have valid syntax."""
    print('\n🔧 Testing Syntax Validation...')

    files_to_test = [
        'graphiti_mcp_server.py',
        'config_manager.py',
        'llm_config.py',
        'embedder_config.py',
        'neo4j_config.py',
        'server_config.py',
        'graphiti_service.py',
        'queue_service.py',
        'entity_types.py',
        'response_types.py',
        'formatting.py',
        'utils.py',
    ]

    success_count = 0

    for file in files_to_test:
        try:
            result = subprocess.run(
                ['python', '-m', 'py_compile', file], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                print(f'   ✅ {file}: Syntax valid')
                success_count += 1
            else:
                print(f'   ❌ {file}: Syntax error - {result.stderr.strip()}')

        except subprocess.TimeoutExpired:
            print(f'   ❌ {file}: Syntax check timeout')
        except Exception as e:
            print(f'   ❌ {file}: Syntax check error - {e}')

    print(f'   📊 Syntax Results: {success_count}/{len(files_to_test)} files valid')
    return success_count == len(files_to_test)


def main():
    """Run the validation tests."""
    print('🧪 Graphiti MCP Server Refactoring Validation')
    print('=' * 55)

    results = {}

    # Test 1: Syntax validation
    results['syntax'] = test_syntax_validation()

    # Test 2: Import validation
    results['imports'] = test_import_validation()

    # Test 3: Server startup
    results['startup'] = test_server_startup()

    # Summary
    print('\n' + '=' * 55)
    print('📊 VALIDATION SUMMARY')
    print('-' * 25)
    print(f'Syntax Validation:    {"✅ PASS" if results["syntax"] else "❌ FAIL"}')
    print(f'Import Validation:    {"✅ PASS" if results["imports"] else "❌ FAIL"}')
    print(f'Startup Validation:   {"✅ PASS" if results["startup"] else "❌ FAIL"}')

    overall_success = all(results.values())
    print('-' * 25)
    print(f'🎯 OVERALL: {"✅ SUCCESS" if overall_success else "❌ FAILED"}')

    if overall_success:
        print('\n🎉 Refactoring validation successful!')
        print('   ✅ All modules have valid syntax')
        print('   ✅ All imports work correctly')
        print('   ✅ Server initializes successfully')
        print('   ✅ The refactored MCP server is ready for use!')
    else:
        print('\n⚠️  Some validation issues detected.')
        print('   Please review the failed tests above.')

    return 0 if overall_success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
