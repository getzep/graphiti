from graphiti_core.helpers import lucene_sanitize


def test_lucene_sanitize():
    # Call the function with test data
    queries = [
        (
            'This has every secape character + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /',
            'This has every secape character \+ \- \&\& \|\| \! \( \) \{ \} \[ \] \^ \\" \~ \* \? \: \\\ \/',
        ),
        ('This has no escape characters', 'This has no escape characters'),
    ]

    for query, assert_result in queries:
        result = lucene_sanitize(query)
        assert assert_result == result
