#!/usr/bin/env python3
"""
Simple validation test for the refactored Graphiti MCP Server.
Tests basic server startup functionality.
"""

import os
import subprocess
import sys
import time


def test_server_startup():
    """Test that the refactored server starts up successfully."""
    print('🚀 Testing Graphiti MCP Server Startup...')

    # Skip server startup test in CI - we have comprehensive integration tests
    if os.environ.get('CI'):
        print(
            '   ⚠️  Skipping server startup test in CI (comprehensive integration tests handle this)'
        )
        return True

    # Check if uv is available
    uv_cmd = None
    for potential_uv in ['uv', '/Users/danielchalef/.local/bin/uv', '/root/.local/bin/uv']:
        try:
            result = subprocess.run([potential_uv, '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                uv_cmd = potential_uv
                break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    if not uv_cmd:
        print('   ⚠️  uv not found in PATH, skipping server startup test')
        return True

    try:
        # Start the server and capture output
        process = subprocess.Popen(
            [uv_cmd, 'run', 'main.py', '--transport', 'stdio'],
            env={
                'NEO4J_URI': 'bolt://localhost:7687',
                'NEO4J_USER': 'neo4j',
                'NEO4J_PASSWORD': 'demodemo',
                'PATH': os.environ.get('PATH', ''),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for initialization and capture output
        captured_output = []
        start_time = time.time()
        server_initialized = False

        # Monitor server output
        while time.time() - start_time < 10:
            try:
                # Check if process has terminated
                if process.poll() is not None:
                    stdout, stderr = process.communicate(timeout=1)
                    captured_output.extend(['   📋 ' + line for line in stdout.split('\n') if line])
                    captured_output.extend(['   📋 ' + line for line in stderr.split('\n') if line])
                    break

                # Check stderr for initialization messages
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    print(f'   📋 {line.strip()}')
                    if 'Starting MCP server' in line or 'Successfully initialized' in line:
                        server_initialized = True
                        break
                    if 'Failed to initialize' in line or 'Error' in line:
                        break

                if server_initialized:
                    break
                time.sleep(0.5)
            except subprocess.TimeoutExpired:
                pass

        # Clean up process
        if process.poll() is None:
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()

        return server_initialized or process.returncode == 0

    except Exception as e:
        print(f'   ⚠️  Timeout waiting for initialization or server startup failed')
        return False


if __name__ == '__main__':
    print('🧪 Graphiti MCP Server Validation')
    print('=' * 55)

    startup_pass = test_server_startup()

    print('\n' + '=' * 55)
    print('📊 VALIDATION SUMMARY')
    print('-------------------------')
    print(f'Startup Validation:   {"✅ PASS" if startup_pass else "❌ FAIL"}')
    print('-------------------------')
    print(f'🎯 OVERALL: {"✅ PASSED" if startup_pass else "❌ FAILED"}')

    if not startup_pass:
        print('\n⚠️  Some validation issues detected.')
        print('   Please review the failed tests above.')
        sys.exit(1)