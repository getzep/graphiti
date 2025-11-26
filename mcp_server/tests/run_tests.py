#!/usr/bin/env python3
"""
Test runner for Graphiti MCP integration tests.
Provides various test execution modes and reporting options.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    # Try loading from current directory
    load_dotenv()


class TestRunner:
    """Orchestrate test execution with various configurations."""

    def __init__(self, args):
        self.args = args
        self.test_dir = Path(__file__).parent
        self.results = {}

    def check_prerequisites(self) -> dict[str, bool]:
        """Check if required services and dependencies are available."""
        checks = {}

        # Check for OpenAI API key if not using mocks
        if not self.args.mock_llm:
            api_key = os.environ.get('OPENAI_API_KEY')
            checks['openai_api_key'] = bool(api_key)
            if not api_key:
                # Check if .env file exists for helpful message
                env_path = Path(__file__).parent.parent / '.env'
                if not env_path.exists():
                    checks['openai_api_key_hint'] = (
                        'Set OPENAI_API_KEY in environment or create mcp_server/.env file'
                    )
        else:
            checks['openai_api_key'] = True

        # Check database availability based on backend
        if self.args.database == 'neo4j':
            checks['neo4j'] = self._check_neo4j()
        elif self.args.database == 'falkordb':
            checks['falkordb'] = self._check_falkordb()

        # Check Python dependencies
        checks['mcp'] = self._check_python_package('mcp')
        checks['pytest'] = self._check_python_package('pytest')
        checks['pytest-asyncio'] = self._check_python_package('pytest-asyncio')

        return checks

    def _check_neo4j(self) -> bool:
        """Check if Neo4j is available."""
        try:
            import neo4j

            # Try to connect
            uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
            user = os.environ.get('NEO4J_USER', 'neo4j')
            password = os.environ.get('NEO4J_PASSWORD', 'graphiti')

            driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                session.run('RETURN 1')
            driver.close()
            return True
        except Exception:
            return False

    def _check_falkordb(self) -> bool:
        """Check if FalkorDB is available."""
        try:
            import redis

            uri = os.environ.get('FALKORDB_URI', 'redis://localhost:6379')
            r = redis.from_url(uri)
            r.ping()
            return True
        except Exception:
            return False

    def _check_python_package(self, package: str) -> bool:
        """Check if a Python package is installed."""
        try:
            __import__(package.replace('-', '_'))
            return True
        except ImportError:
            return False

    def run_test_suite(self, suite: str) -> int:
        """Run a specific test suite."""
        pytest_args = ['-v', '--tb=short']

        # Add database marker
        if self.args.database:
            for db in ['neo4j', 'falkordb']:
                if db != self.args.database:
                    pytest_args.extend(['-m', f'not requires_{db}'])

        # Add suite-specific arguments
        if suite == 'unit':
            pytest_args.extend(['-m', 'unit', 'test_*.py'])
        elif suite == 'integration':
            pytest_args.extend(['-m', 'integration or not unit', 'test_*.py'])
        elif suite == 'comprehensive':
            pytest_args.append('test_comprehensive_integration.py')
        elif suite == 'async':
            pytest_args.append('test_async_operations.py')
        elif suite == 'stress':
            pytest_args.extend(['-m', 'slow', 'test_stress_load.py'])
        elif suite == 'smoke':
            # Quick smoke test - just basic operations
            pytest_args.extend(
                [
                    'test_comprehensive_integration.py::TestCoreOperations::test_server_initialization',
                    'test_comprehensive_integration.py::TestCoreOperations::test_add_text_memory',
                ]
            )
        elif suite == 'all':
            pytest_args.append('.')
        else:
            pytest_args.append(suite)

        # Add coverage if requested
        if self.args.coverage:
            pytest_args.extend(['--cov=../src', '--cov-report=html'])

        # Add parallel execution if requested
        if self.args.parallel:
            pytest_args.extend(['-n', str(self.args.parallel)])

        # Add verbosity
        if self.args.verbose:
            pytest_args.append('-vv')

        # Add markers to skip
        if self.args.skip_slow:
            pytest_args.extend(['-m', 'not slow'])

        # Add timeout override
        if self.args.timeout:
            pytest_args.extend(['--timeout', str(self.args.timeout)])

        # Add environment variables
        env = os.environ.copy()
        if self.args.mock_llm:
            env['USE_MOCK_LLM'] = 'true'
        if self.args.database:
            env['DATABASE_PROVIDER'] = self.args.database

        # Run tests from the test directory
        print(f'Running {suite} tests with pytest args: {" ".join(pytest_args)}')

        # Change to test directory to run tests
        original_dir = os.getcwd()
        os.chdir(self.test_dir)

        try:
            result = pytest.main(pytest_args)
        finally:
            os.chdir(original_dir)

        return result

    def run_performance_benchmark(self):
        """Run performance benchmarking suite."""
        print('Running performance benchmarks...')

        # Import test modules

        # Run performance tests
        result = pytest.main(
            [
                '-v',
                'test_comprehensive_integration.py::TestPerformance',
                'test_async_operations.py::TestAsyncPerformance',
                '--benchmark-only' if self.args.benchmark_only else '',
            ]
        )

        return result

    def generate_report(self):
        """Generate test execution report."""
        report = []
        report.append('\n' + '=' * 60)
        report.append('GRAPHITI MCP TEST EXECUTION REPORT')
        report.append('=' * 60)

        # Prerequisites check
        checks = self.check_prerequisites()
        report.append('\nPrerequisites:')
        for check, passed in checks.items():
            status = '✅' if passed else '❌'
            report.append(f'  {status} {check}')

        # Test configuration
        report.append('\nConfiguration:')
        report.append(f'  Database: {self.args.database}')
        report.append(f'  Mock LLM: {self.args.mock_llm}')
        report.append(f'  Parallel: {self.args.parallel or "No"}')
        report.append(f'  Timeout: {self.args.timeout}s')

        # Results summary (if available)
        if self.results:
            report.append('\nResults:')
            for suite, result in self.results.items():
                status = '✅ Passed' if result == 0 else f'❌ Failed ({result})'
                report.append(f'  {suite}: {status}')

        report.append('=' * 60)
        return '\n'.join(report)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Run Graphiti MCP integration tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Suites:
  unit          - Run unit tests only
  integration   - Run integration tests
  comprehensive - Run comprehensive integration test suite
  async         - Run async operation tests
  stress        - Run stress and load tests
  smoke         - Run quick smoke tests
  all           - Run all tests

Examples:
  python run_tests.py smoke                    # Quick smoke test
  python run_tests.py integration --parallel 4 # Run integration tests in parallel
  python run_tests.py stress --database neo4j  # Run stress tests with Neo4j
  python run_tests.py all --coverage          # Run all tests with coverage
        """,
    )

    parser.add_argument(
        'suite',
        choices=['unit', 'integration', 'comprehensive', 'async', 'stress', 'smoke', 'all'],
        help='Test suite to run',
    )

    parser.add_argument(
        '--database',
        choices=['neo4j', 'falkordb'],
        default='falkordb',
        help='Database backend to test (default: falkordb)',
    )

    parser.add_argument('--mock-llm', action='store_true', help='Use mock LLM for faster testing')

    parser.add_argument(
        '--parallel', type=int, metavar='N', help='Run tests in parallel with N workers'
    )

    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')

    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    parser.add_argument('--skip-slow', action='store_true', help='Skip slow tests')

    parser.add_argument(
        '--timeout', type=int, default=300, help='Test timeout in seconds (default: 300)'
    )

    parser.add_argument('--benchmark-only', action='store_true', help='Run only benchmark tests')

    parser.add_argument(
        '--check-only', action='store_true', help='Only check prerequisites without running tests'
    )

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner(args)

    # Check prerequisites
    if args.check_only:
        print(runner.generate_report())
        sys.exit(0)

    # Check if prerequisites are met
    checks = runner.check_prerequisites()
    # Filter out hint keys from validation
    validation_checks = {k: v for k, v in checks.items() if not k.endswith('_hint')}

    if not all(validation_checks.values()):
        print('⚠️  Some prerequisites are not met:')
        for check, passed in checks.items():
            if check.endswith('_hint'):
                continue  # Skip hint entries
            if not passed:
                print(f'  ❌ {check}')
                # Show hint if available
                hint_key = f'{check}_hint'
                if hint_key in checks:
                    print(f'     💡 {checks[hint_key]}')

        if not args.mock_llm and not checks.get('openai_api_key'):
            print('\n💡 Tip: Use --mock-llm to run tests without OpenAI API key')

        response = input('\nContinue anyway? (y/N): ')
        if response.lower() != 'y':
            sys.exit(1)

    # Run tests
    print(f'\n🚀 Starting test execution: {args.suite}')
    start_time = time.time()

    if args.benchmark_only:
        result = runner.run_performance_benchmark()
    else:
        result = runner.run_test_suite(args.suite)

    duration = time.time() - start_time

    # Store results
    runner.results[args.suite] = result

    # Generate and print report
    print(runner.generate_report())
    print(f'\n⏱️  Test execution completed in {duration:.2f} seconds')

    # Exit with test result code
    sys.exit(result)


if __name__ == '__main__':
    main()
