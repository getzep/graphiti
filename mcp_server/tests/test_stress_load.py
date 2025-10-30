#!/usr/bin/env python3
"""
Stress and load testing for Graphiti MCP Server.
Tests system behavior under high load, resource constraints, and edge conditions.
"""

import asyncio
import gc
import random
import time
from dataclasses import dataclass

import psutil
import pytest
from test_fixtures import TestDataGenerator, graphiti_test_client


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""

    num_clients: int = 10
    operations_per_client: int = 100
    ramp_up_time: float = 5.0  # seconds
    test_duration: float = 60.0  # seconds
    target_throughput: float | None = None  # ops/sec
    think_time: float = 0.1  # seconds between ops


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_operations: int
    successful_operations: int
    failed_operations: int
    duration: float
    throughput: float
    average_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    errors: dict[str, int]
    resource_usage: dict[str, float]


class LoadTester:
    """Orchestrate load testing scenarios."""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.metrics: list[tuple[float, float, bool]] = []  # (start, duration, success)
        self.errors: dict[str, int] = {}
        self.start_time: float | None = None

    async def run_client_workload(self, client_id: int, session, group_id: str) -> dict[str, int]:
        """Run workload for a single simulated client."""
        stats = {'success': 0, 'failure': 0}
        data_gen = TestDataGenerator()

        # Ramp-up delay
        ramp_delay = (client_id / self.config.num_clients) * self.config.ramp_up_time
        await asyncio.sleep(ramp_delay)

        for op_num in range(self.config.operations_per_client):
            operation_start = time.time()

            try:
                # Randomly select operation type
                operation = random.choice(
                    [
                        'add_memory',
                        'search_memory_nodes',
                        'get_episodes',
                    ]
                )

                if operation == 'add_memory':
                    args = {
                        'name': f'Load Test {client_id}-{op_num}',
                        'episode_body': data_gen.generate_technical_document(),
                        'source': 'text',
                        'source_description': 'load test',
                        'group_id': group_id,
                    }
                elif operation == 'search_memory_nodes':
                    args = {
                        'query': random.choice(['performance', 'architecture', 'test', 'data']),
                        'group_id': group_id,
                        'limit': 10,
                    }
                else:  # get_episodes
                    args = {
                        'group_id': group_id,
                        'last_n': 10,
                    }

                # Execute operation with timeout
                await asyncio.wait_for(session.call_tool(operation, args), timeout=30.0)

                duration = time.time() - operation_start
                self.metrics.append((operation_start, duration, True))
                stats['success'] += 1

            except asyncio.TimeoutError:
                duration = time.time() - operation_start
                self.metrics.append((operation_start, duration, False))
                self.errors['timeout'] = self.errors.get('timeout', 0) + 1
                stats['failure'] += 1

            except Exception as e:
                duration = time.time() - operation_start
                self.metrics.append((operation_start, duration, False))
                error_type = type(e).__name__
                self.errors[error_type] = self.errors.get(error_type, 0) + 1
                stats['failure'] += 1

            # Think time between operations
            await asyncio.sleep(self.config.think_time)

            # Stop if we've exceeded test duration
            if self.start_time and (time.time() - self.start_time) > self.config.test_duration:
                break

        return stats

    def calculate_results(self) -> LoadTestResult:
        """Calculate load test results from metrics."""
        if not self.metrics:
            return LoadTestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {})

        successful = [m for m in self.metrics if m[2]]
        failed = [m for m in self.metrics if not m[2]]

        latencies = sorted([m[1] for m in self.metrics])
        duration = max([m[0] + m[1] for m in self.metrics]) - min([m[0] for m in self.metrics])

        # Calculate percentiles
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        # Get resource usage
        process = psutil.Process()
        resource_usage = {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'num_threads': process.num_threads(),
        }

        return LoadTestResult(
            total_operations=len(self.metrics),
            successful_operations=len(successful),
            failed_operations=len(failed),
            duration=duration,
            throughput=len(self.metrics) / duration if duration > 0 else 0,
            average_latency=sum(latencies) / len(latencies) if latencies else 0,
            p50_latency=percentile(latencies, 50),
            p95_latency=percentile(latencies, 95),
            p99_latency=percentile(latencies, 99),
            max_latency=max(latencies) if latencies else 0,
            errors=self.errors,
            resource_usage=resource_usage,
        )


class TestLoadScenarios:
    """Various load testing scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load(self):
        """Test system under sustained moderate load."""
        config = LoadTestConfig(
            num_clients=5,
            operations_per_client=20,
            ramp_up_time=2.0,
            test_duration=30.0,
            think_time=0.5,
        )

        async with graphiti_test_client() as (session, group_id):
            tester = LoadTester(config)
            tester.start_time = time.time()

            # Run client workloads
            client_tasks = []
            for client_id in range(config.num_clients):
                task = tester.run_client_workload(client_id, session, group_id)
                client_tasks.append(task)

            # Execute all clients
            await asyncio.gather(*client_tasks)

            # Calculate results
            results = tester.calculate_results()

            # Assertions
            assert results.successful_operations > results.failed_operations
            assert results.average_latency < 5.0, (
                f'Average latency too high: {results.average_latency:.2f}s'
            )
            assert results.p95_latency < 10.0, f'P95 latency too high: {results.p95_latency:.2f}s'

            # Report results
            print('\nSustained Load Test Results:')
            print(f'  Total operations: {results.total_operations}')
            print(
                f'  Success rate: {results.successful_operations / results.total_operations * 100:.1f}%'
            )
            print(f'  Throughput: {results.throughput:.2f} ops/s')
            print(f'  Avg latency: {results.average_latency:.2f}s')
            print(f'  P95 latency: {results.p95_latency:.2f}s')

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_spike_load(self):
        """Test system response to sudden load spikes."""
        async with graphiti_test_client() as (session, group_id):
            # Normal load phase
            normal_tasks = []
            for i in range(3):
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': f'Normal Load {i}',
                        'episode_body': 'Normal operation',
                        'source': 'text',
                        'source_description': 'normal',
                        'group_id': group_id,
                    },
                )
                normal_tasks.append(task)
                await asyncio.sleep(0.5)

            await asyncio.gather(*normal_tasks)

            # Spike phase - sudden burst of requests
            spike_start = time.time()
            spike_tasks = []
            for i in range(50):
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': f'Spike Load {i}',
                        'episode_body': TestDataGenerator.generate_technical_document(),
                        'source': 'text',
                        'source_description': 'spike',
                        'group_id': group_id,
                    },
                )
                spike_tasks.append(task)

            # Execute spike
            spike_results = await asyncio.gather(*spike_tasks, return_exceptions=True)
            spike_duration = time.time() - spike_start

            # Analyze spike handling
            spike_failures = sum(1 for r in spike_results if isinstance(r, Exception))
            spike_success_rate = (len(spike_results) - spike_failures) / len(spike_results)

            print('\nSpike Load Test Results:')
            print(f'  Spike size: {len(spike_tasks)} operations')
            print(f'  Duration: {spike_duration:.2f}s')
            print(f'  Success rate: {spike_success_rate * 100:.1f}%')
            print(f'  Throughput: {len(spike_tasks) / spike_duration:.2f} ops/s')

            # System should handle at least 80% of spike
            assert spike_success_rate > 0.8, f'Too many failures during spike: {spike_failures}'

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        async with graphiti_test_client() as (session, group_id):
            process = psutil.Process()
            gc.collect()  # Force garbage collection
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform many operations
            for batch in range(10):
                batch_tasks = []
                for i in range(10):
                    task = session.call_tool(
                        'add_memory',
                        {
                            'name': f'Memory Test {batch}-{i}',
                            'episode_body': TestDataGenerator.generate_technical_document(),
                            'source': 'text',
                            'source_description': 'memory test',
                            'group_id': group_id,
                        },
                    )
                    batch_tasks.append(task)

                await asyncio.gather(*batch_tasks)

                # Force garbage collection between batches
                gc.collect()
                await asyncio.sleep(1)

            # Check memory after operations
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - initial_memory

            print('\nMemory Leak Test:')
            print(f'  Initial memory: {initial_memory:.1f} MB')
            print(f'  Final memory: {final_memory:.1f} MB')
            print(f'  Growth: {memory_growth:.1f} MB')

            # Allow for some memory growth but flag potential leaks
            # This is a soft check - actual threshold depends on system
            if memory_growth > 100:  # More than 100MB growth
                print(f'  ⚠️  Potential memory leak detected: {memory_growth:.1f} MB growth')

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_connection_pool_exhaustion(self):
        """Test behavior when connection pools are exhausted."""
        async with graphiti_test_client() as (session, group_id):
            # Create many concurrent long-running operations
            long_tasks = []
            for i in range(100):  # Many more than typical pool size
                task = session.call_tool(
                    'search_memory_nodes',
                    {
                        'query': f'complex query {i} '
                        + ' '.join([TestDataGenerator.fake.word() for _ in range(10)]),
                        'group_id': group_id,
                        'limit': 100,
                    },
                )
                long_tasks.append(task)

            # Execute with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*long_tasks, return_exceptions=True), timeout=60.0
                )

                # Count connection-related errors
                connection_errors = sum(
                    1
                    for r in results
                    if isinstance(r, Exception) and 'connection' in str(r).lower()
                )

                print('\nConnection Pool Test:')
                print(f'  Total requests: {len(long_tasks)}')
                print(f'  Connection errors: {connection_errors}')

            except asyncio.TimeoutError:
                print('  Test timed out - possible deadlock or exhaustion')

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_gradual_degradation(self):
        """Test system degradation under increasing load."""
        async with graphiti_test_client() as (session, group_id):
            load_levels = [5, 10, 20, 40, 80]  # Increasing concurrent operations
            results_by_level = {}

            for level in load_levels:
                level_start = time.time()
                tasks = []

                for i in range(level):
                    task = session.call_tool(
                        'add_memory',
                        {
                            'name': f'Load Level {level} Op {i}',
                            'episode_body': f'Testing at load level {level}',
                            'source': 'text',
                            'source_description': 'degradation test',
                            'group_id': group_id,
                        },
                    )
                    tasks.append(task)

                # Execute level
                level_results = await asyncio.gather(*tasks, return_exceptions=True)
                level_duration = time.time() - level_start

                # Calculate metrics
                failures = sum(1 for r in level_results if isinstance(r, Exception))
                success_rate = (level - failures) / level * 100
                throughput = level / level_duration

                results_by_level[level] = {
                    'success_rate': success_rate,
                    'throughput': throughput,
                    'duration': level_duration,
                }

                print(f'\nLoad Level {level}:')
                print(f'  Success rate: {success_rate:.1f}%')
                print(f'  Throughput: {throughput:.2f} ops/s')
                print(f'  Duration: {level_duration:.2f}s')

                # Brief pause between levels
                await asyncio.sleep(2)

            # Verify graceful degradation
            # Success rate should not drop below 50% even at high load
            for level, metrics in results_by_level.items():
                assert metrics['success_rate'] > 50, f'Poor performance at load level {level}'


class TestResourceLimits:
    """Test behavior at resource limits."""

    @pytest.mark.asyncio
    async def test_large_payload_handling(self):
        """Test handling of very large payloads."""
        async with graphiti_test_client() as (session, group_id):
            payload_sizes = [
                (1_000, '1KB'),
                (10_000, '10KB'),
                (100_000, '100KB'),
                (1_000_000, '1MB'),
            ]

            for size, label in payload_sizes:
                content = 'x' * size

                start_time = time.time()
                try:
                    await asyncio.wait_for(
                        session.call_tool(
                            'add_memory',
                            {
                                'name': f'Large Payload {label}',
                                'episode_body': content,
                                'source': 'text',
                                'source_description': 'payload test',
                                'group_id': group_id,
                            },
                        ),
                        timeout=30.0,
                    )
                    duration = time.time() - start_time
                    status = '✅ Success'

                except asyncio.TimeoutError:
                    duration = 30.0
                    status = '⏱️  Timeout'

                except Exception as e:
                    duration = time.time() - start_time
                    status = f'❌ Error: {type(e).__name__}'

                print(f'Payload {label}: {status} ({duration:.2f}s)')

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test handling of rate limits."""
        async with graphiti_test_client() as (session, group_id):
            # Rapid fire requests to trigger rate limits
            rapid_tasks = []
            for i in range(100):
                task = session.call_tool(
                    'add_memory',
                    {
                        'name': f'Rate Limit Test {i}',
                        'episode_body': f'Testing rate limit {i}',
                        'source': 'text',
                        'source_description': 'rate test',
                        'group_id': group_id,
                    },
                )
                rapid_tasks.append(task)

            # Execute without delays
            results = await asyncio.gather(*rapid_tasks, return_exceptions=True)

            # Count rate limit errors
            rate_limit_errors = sum(
                1
                for r in results
                if isinstance(r, Exception) and ('rate' in str(r).lower() or '429' in str(r))
            )

            print('\nRate Limit Test:')
            print(f'  Total requests: {len(rapid_tasks)}')
            print(f'  Rate limit errors: {rate_limit_errors}')
            print(
                f'  Success rate: {(len(rapid_tasks) - rate_limit_errors) / len(rapid_tasks) * 100:.1f}%'
            )


def generate_load_test_report(results: list[LoadTestResult]) -> str:
    """Generate comprehensive load test report."""
    report = []
    report.append('\n' + '=' * 60)
    report.append('LOAD TEST REPORT')
    report.append('=' * 60)

    for i, result in enumerate(results):
        report.append(f'\nTest Run {i + 1}:')
        report.append(f'  Total Operations: {result.total_operations}')
        report.append(
            f'  Success Rate: {result.successful_operations / result.total_operations * 100:.1f}%'
        )
        report.append(f'  Throughput: {result.throughput:.2f} ops/s')
        report.append(
            f'  Latency (avg/p50/p95/p99/max): {result.average_latency:.2f}/{result.p50_latency:.2f}/{result.p95_latency:.2f}/{result.p99_latency:.2f}/{result.max_latency:.2f}s'
        )

        if result.errors:
            report.append('  Errors:')
            for error_type, count in result.errors.items():
                report.append(f'    {error_type}: {count}')

        report.append('  Resource Usage:')
        for metric, value in result.resource_usage.items():
            report.append(f'    {metric}: {value:.2f}')

    report.append('=' * 60)
    return '\n'.join(report)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto', '-m', 'slow'])
