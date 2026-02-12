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

from dataclasses import dataclass
from threading import Lock


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class PromptTokenUsage:
    """Accumulated token usage for a specific prompt type."""

    prompt_name: str
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_input_tokens(self) -> float:
        return self.total_input_tokens / self.call_count if self.call_count > 0 else 0

    @property
    def avg_output_tokens(self) -> float:
        return self.total_output_tokens / self.call_count if self.call_count > 0 else 0


class TokenUsageTracker:
    """Thread-safe tracker for LLM token usage by prompt type."""

    def __init__(self):
        self._usage: dict[str, PromptTokenUsage] = {}
        self._lock = Lock()

    def record(self, prompt_name: str | None, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for a prompt.

        Args:
            prompt_name: Name of the prompt (e.g., 'extract_nodes.extract_message')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
        """
        key = prompt_name or 'unknown'

        with self._lock:
            if key not in self._usage:
                self._usage[key] = PromptTokenUsage(prompt_name=key)

            self._usage[key].call_count += 1
            self._usage[key].total_input_tokens += input_tokens
            self._usage[key].total_output_tokens += output_tokens

    def get_usage(self) -> dict[str, PromptTokenUsage]:
        """Get a copy of current token usage by prompt type."""
        with self._lock:
            return {k: PromptTokenUsage(
                prompt_name=v.prompt_name,
                call_count=v.call_count,
                total_input_tokens=v.total_input_tokens,
                total_output_tokens=v.total_output_tokens,
            ) for k, v in self._usage.items()}

    def get_total_usage(self) -> TokenUsage:
        """Get total token usage across all prompts."""
        with self._lock:
            total_input = sum(u.total_input_tokens for u in self._usage.values())
            total_output = sum(u.total_output_tokens for u in self._usage.values())
            return TokenUsage(input_tokens=total_input, output_tokens=total_output)

    def reset(self) -> None:
        """Reset all tracked usage."""
        with self._lock:
            self._usage.clear()

    def print_summary(self, sort_by: str = 'total_tokens') -> None:
        """Print a formatted summary of token usage.

        Args:
            sort_by: Sort key - 'total_tokens', 'input_tokens', 'output_tokens', 'call_count', or 'prompt_name'
        """
        usage = self.get_usage()
        if not usage:
            print('No token usage recorded.')
            return

        # Sort usage
        sort_keys = {
            'total_tokens': lambda x: x[1].total_tokens,
            'input_tokens': lambda x: x[1].total_input_tokens,
            'output_tokens': lambda x: x[1].total_output_tokens,
            'call_count': lambda x: x[1].call_count,
            'prompt_name': lambda x: x[0],
        }
        sort_fn = sort_keys.get(sort_by, sort_keys['total_tokens'])
        sorted_usage = sorted(usage.items(), key=sort_fn, reverse=(sort_by != 'prompt_name'))

        # Print header
        print('\n' + '=' * 100)
        print('TOKEN USAGE SUMMARY')
        print('=' * 100)
        print(
            f'{"Prompt Type":<45} {"Calls":>8} {"Input":>12} {"Output":>12} {"Total":>12} {"Avg In":>10} {"Avg Out":>10}'
        )
        print('-' * 100)

        # Print each prompt's usage
        for prompt_name, prompt_usage in sorted_usage:
            print(
                f'{prompt_name:<45} {prompt_usage.call_count:>8} {prompt_usage.total_input_tokens:>12,} '
                f'{prompt_usage.total_output_tokens:>12,} {prompt_usage.total_tokens:>12,} '
                f'{prompt_usage.avg_input_tokens:>10,.1f} {prompt_usage.avg_output_tokens:>10,.1f}'
            )

        # Print totals
        total = self.get_total_usage()
        total_calls = sum(u.call_count for u in usage.values())
        print('-' * 100)
        print(
            f'{"TOTAL":<45} {total_calls:>8} {total.input_tokens:>12,} '
            f'{total.output_tokens:>12,} {total.total_tokens:>12,}'
        )
        print('=' * 100 + '\n')
