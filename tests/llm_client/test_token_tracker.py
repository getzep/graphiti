"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from concurrent.futures import ThreadPoolExecutor

from graphiti_core.llm_client.token_tracker import (
    PromptTokenUsage,
    TokenUsage,
    TokenUsageTracker,
)


class TestTokenUsage:
    def test_total_tokens(self):
        """Test that total_tokens correctly sums input and output tokens."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_default_values(self):
        """Test that default values are zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0


class TestPromptTokenUsage:
    def test_total_tokens(self):
        """Test that total_tokens correctly sums input and output tokens."""
        usage = PromptTokenUsage(
            prompt_name='test',
            call_count=5,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert usage.total_tokens == 1500

    def test_avg_input_tokens(self):
        """Test average input tokens calculation."""
        usage = PromptTokenUsage(
            prompt_name='test',
            call_count=4,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert usage.avg_input_tokens == 250.0

    def test_avg_output_tokens(self):
        """Test average output tokens calculation."""
        usage = PromptTokenUsage(
            prompt_name='test',
            call_count=4,
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert usage.avg_output_tokens == 125.0

    def test_avg_tokens_zero_calls(self):
        """Test that average returns 0 when call_count is zero."""
        usage = PromptTokenUsage(
            prompt_name='test',
            call_count=0,
            total_input_tokens=0,
            total_output_tokens=0,
        )
        assert usage.avg_input_tokens == 0
        assert usage.avg_output_tokens == 0


class TestTokenUsageTracker:
    def test_record_new_prompt(self):
        """Test recording usage for a new prompt."""
        tracker = TokenUsageTracker()
        tracker.record('extract_nodes', 100, 50)

        usage = tracker.get_usage()
        assert 'extract_nodes' in usage
        assert usage['extract_nodes'].call_count == 1
        assert usage['extract_nodes'].total_input_tokens == 100
        assert usage['extract_nodes'].total_output_tokens == 50

    def test_record_existing_prompt(self):
        """Test that multiple calls accumulate correctly."""
        tracker = TokenUsageTracker()
        tracker.record('extract_nodes', 100, 50)
        tracker.record('extract_nodes', 200, 100)

        usage = tracker.get_usage()
        assert usage['extract_nodes'].call_count == 2
        assert usage['extract_nodes'].total_input_tokens == 300
        assert usage['extract_nodes'].total_output_tokens == 150

    def test_record_none_prompt_name(self):
        """Test that None prompt_name is recorded as 'unknown'."""
        tracker = TokenUsageTracker()
        tracker.record(None, 100, 50)

        usage = tracker.get_usage()
        assert 'unknown' in usage
        assert usage['unknown'].call_count == 1

    def test_record_multiple_prompts(self):
        """Test recording usage for multiple different prompts."""
        tracker = TokenUsageTracker()
        tracker.record('extract_nodes', 100, 50)
        tracker.record('dedupe_nodes', 200, 100)
        tracker.record('extract_edges', 150, 75)

        usage = tracker.get_usage()
        assert len(usage) == 3
        assert 'extract_nodes' in usage
        assert 'dedupe_nodes' in usage
        assert 'extract_edges' in usage

    def test_get_usage_returns_copy(self):
        """Test that get_usage returns a copy, not the internal dict."""
        tracker = TokenUsageTracker()
        tracker.record('test', 100, 50)

        usage1 = tracker.get_usage()
        usage1['test'].total_input_tokens = 9999

        usage2 = tracker.get_usage()
        assert usage2['test'].total_input_tokens == 100  # Original unchanged

    def test_get_total_usage(self):
        """Test getting total usage across all prompts."""
        tracker = TokenUsageTracker()
        tracker.record('extract_nodes', 100, 50)
        tracker.record('dedupe_nodes', 200, 100)
        tracker.record('extract_edges', 150, 75)

        total = tracker.get_total_usage()
        assert total.input_tokens == 450
        assert total.output_tokens == 225
        assert total.total_tokens == 675

    def test_get_total_usage_empty(self):
        """Test getting total usage when no records exist."""
        tracker = TokenUsageTracker()
        total = tracker.get_total_usage()
        assert total.input_tokens == 0
        assert total.output_tokens == 0

    def test_reset(self):
        """Test that reset clears all tracked usage."""
        tracker = TokenUsageTracker()
        tracker.record('extract_nodes', 100, 50)
        tracker.record('dedupe_nodes', 200, 100)

        tracker.reset()

        usage = tracker.get_usage()
        assert len(usage) == 0

        total = tracker.get_total_usage()
        assert total.total_tokens == 0

    def test_thread_safety(self):
        """Test that concurrent access from multiple threads is safe."""
        tracker = TokenUsageTracker()
        num_threads = 10
        calls_per_thread = 100

        def record_tokens(thread_id):
            for _ in range(calls_per_thread):
                tracker.record(f'prompt_{thread_id}', 10, 5)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_tokens, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        usage = tracker.get_usage()
        assert len(usage) == num_threads

        total = tracker.get_total_usage()
        expected_input = num_threads * calls_per_thread * 10
        expected_output = num_threads * calls_per_thread * 5
        assert total.input_tokens == expected_input
        assert total.output_tokens == expected_output

    def test_concurrent_same_prompt(self):
        """Test concurrent access to the same prompt name."""
        tracker = TokenUsageTracker()
        num_threads = 10
        calls_per_thread = 100

        def record_tokens():
            for _ in range(calls_per_thread):
                tracker.record('shared_prompt', 10, 5)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_tokens) for _ in range(num_threads)]
            for f in futures:
                f.result()

        usage = tracker.get_usage()
        assert usage['shared_prompt'].call_count == num_threads * calls_per_thread
        assert usage['shared_prompt'].total_input_tokens == num_threads * calls_per_thread * 10
        assert usage['shared_prompt'].total_output_tokens == num_threads * calls_per_thread * 5
