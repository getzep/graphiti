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

import pytest

from graphiti_core.helpers import lucene_sanitize


def test_lucene_sanitize():
    # Call the function with test data
    queries = [
        (
            'This has every escape character + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            '\\This has every escape character \\+ \\- \\&\\& \\|\\| \\! \\( \\) \\{ \\} \\[ \\] \\^ \\" \\~ \\* \\? \\: \\\\ \\/',
        ),
        ('this has no escape characters', 'this has no escape characters'),
    ]

    for query, assert_result in queries:
        result = lucene_sanitize(query)
        assert assert_result == result


if __name__ == '__main__':
    pytest.main([__file__])
