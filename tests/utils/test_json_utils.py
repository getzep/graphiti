"""Unit tests for graphiti_core.utils.json_utils.safe_json_loads."""

import json

import pytest

from graphiti_core.utils.json_utils import safe_json_loads

# ── 1. Valid JSON (pass-through, no fix needed) ──────────────────────


class TestValidJson:
    """safe_json_loads must behave identically to json.loads for valid input."""

    def test_simple_object(self):
        assert safe_json_loads('{"a": 1}') == {'a': 1}

    def test_nested_object(self):
        raw = '{"a": {"b": [1, 2, 3]}, "c": true}'
        assert safe_json_loads(raw) == json.loads(raw)

    def test_array(self):
        assert safe_json_loads('[1, 2, 3]') == [1, 2, 3]

    def test_empty_object(self):
        assert safe_json_loads('{}') == {}

    def test_empty_array(self):
        assert safe_json_loads('[]') == []

    def test_string_value(self):
        assert safe_json_loads('"hello"') == 'hello'

    def test_number_value(self):
        assert safe_json_loads('42') == 42

    def test_null_value(self):
        assert safe_json_loads('null') is None

    def test_boolean_values(self):
        assert safe_json_loads('true') is True
        assert safe_json_loads('false') is False

    def test_unicode_content(self):
        raw = '{"name": "Schr\\u00f6dinger"}'
        assert safe_json_loads(raw) == {'name': 'Schrödinger'}

    def test_valid_escape_sequences_preserved(self):
        """All 8 JSON escape sequences must pass through unchanged."""
        raw = r'{"a": "line1\nline2\ttab", "b": "quote\"slash\\slash\/", "c": "\b\f\r"}'
        result = safe_json_loads(raw)
        assert result == json.loads(raw)

    def test_multiline_json(self):
        raw = '{\n  "name": "test",\n  "value": 42\n}'
        assert safe_json_loads(raw) == {'name': 'test', 'value': 42}


# ── 2. LaTeX in entity names (the core issue #1204) ─────────────────


class TestLatexBackslashes:
    """LaTeX commands that are NOT valid JSON escapes must be fixed."""

    def test_hat(self):
        raw = r'{"name": "operator $\hat{H}$"}'
        result = safe_json_loads(raw)
        assert result['name'] == 'operator $\\hat{H}$'

    def test_psi(self):
        raw = r'{"name": "wave function $\psi$"}'
        result = safe_json_loads(raw)
        assert result['name'] == 'wave function $\\psi$'

    def test_partial(self):
        raw = r'{"name": "$\partial x$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\partial x$'

    def test_mathbf(self):
        raw = r'{"name": "vector $\mathbf{v}$"}'
        result = safe_json_loads(raw)
        assert result['name'] == 'vector $\\mathbf{v}$'

    def test_alpha_beta_gamma(self):
        raw = r'{"name": "$\alpha + \beta = \gamma$"}'
        result = safe_json_loads(raw)
        # \b is a valid JSON escape (backspace), so \beta becomes \x08 + 'eta'
        # \a and \g are invalid → fixed to \\a, \\g
        assert '\\alpha' in result['name']
        assert '\\gamma' in result['name']
        # \beta silently corrupted: \b → backspace char
        assert '\x08eta' in result['name']

    def test_sum_and_int(self):
        raw = r'{"name": "$\sum_{i} \int_{0}^{1} dx$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\sum_{i} \\int_{0}^{1} dx$'

    def test_sqrt(self):
        raw = r'{"name": "$\sqrt{x^2 + y^2}$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\sqrt{x^2 + y^2}$'

    def test_overline(self):
        raw = r'{"name": "$\overline{AB}$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\overline{AB}$'

    def test_lambda_and_omega(self):
        raw = r'{"name": "$\lambda \omega$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\lambda \\omega$'

    def test_mathrm(self):
        raw = r'{"name": "$\mathrm{pH}$"}'
        result = safe_json_loads(raw)
        assert result['name'] == '$\\mathrm{pH}$'


# ── 3. Complex entity extraction payloads ────────────────────────────


class TestComplexPayloads:
    """Simulated LLM extraction outputs with multiple entities."""

    def test_schrodinger_extraction(self):
        raw = (
            r'{"extracted_entities": ['
            r'{"name": "Schrödinger equation $i\hbar\partial_t \psi = \hat{H}\psi$", '
            r'"entity_type_id": 0},'
            r'{"name": "quantum mechanics", "entity_type_id": 1}'
            r']}'
        )
        result = safe_json_loads(raw)
        entities = result['extracted_entities']
        assert len(entities) == 2
        assert 'Schrödinger' in entities[0]['name']
        assert '\\hat{H}' in entities[0]['name']
        assert entities[1]['name'] == 'quantum mechanics'

    def test_maxwell_equations(self):
        raw = (
            r'{"facts": ['
            r'{"content": "$\vec{E}$ and $\vec{B}$ are coupled via Maxwell equations",'
            r' "entity_type": "physics_law"}'
            r']}'
        )
        result = safe_json_loads(raw)
        assert '\\vec{E}' in result['facts'][0]['content']

    def test_mixed_clean_and_latex_entities(self):
        raw = (
            r'{"entities": ['
            r'{"name": "Isaac Newton"},'
            r'{"name": "$\vec{F} = m\vec{a}$"},'
            r'{"name": "classical mechanics"}'
            r']}'
        )
        result = safe_json_loads(raw)
        assert result['entities'][0]['name'] == 'Isaac Newton'
        assert '\\vec{F}' in result['entities'][1]['name']
        assert result['entities'][2]['name'] == 'classical mechanics'


# ── 4. Windows paths (another backslash source) ─────────────────────


class TestWindowsPaths:
    """Windows-style paths can also introduce illegal escapes."""

    def test_windows_path_with_users(self):
        # \U is not a valid JSON escape (only lowercase \u is)
        raw = r'{"path": "C:\Users\documents\report.pdf"}'
        result = safe_json_loads(raw)
        assert 'Users' in result['path']
        assert 'documents' in result['path']

    def test_windows_path_with_program_files(self):
        raw = r'{"path": "C:\Program Files\app\config.ini"}'
        result = safe_json_loads(raw)
        assert 'Program Files' in result['path']


# ── 5. Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary conditions and tricky inputs."""

    def test_double_backslash_preserved(self):
        """Already-escaped backslashes must NOT be double-escaped."""
        raw = r'{"name": "already\\escaped"}'
        result = safe_json_loads(raw)
        assert result['name'] == 'already\\escaped'

    def test_backslash_at_end_still_raises(self):
        """A trailing lone backslash is still invalid and should raise."""
        raw = '{"name": "trailing\\'
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads(raw)

    def test_truly_broken_json_still_raises(self):
        """Non-backslash JSON errors must still raise."""
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads('{missing quotes: value}')

    def test_truncated_json_still_raises(self):
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads('{"name": "incom')

    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads('')

    def test_multiple_illegal_escapes_in_one_value(self):
        raw = r'{"eq": "$\hat{H}\psi = E\psi$"}'
        result = safe_json_loads(raw)
        assert '\\hat{H}' in result['eq']
        assert '\\psi' in result['eq']

    def test_backslash_before_digit(self):
        """\\1, \\2 etc. are not valid JSON escapes."""
        raw = r'{"ref": "see section \1 and \2"}'
        result = safe_json_loads(raw)
        assert '\\1' in result['ref']
        assert '\\2' in result['ref']

    def test_backslash_before_space(self):
        raw = r'{"v": "a\ b"}'
        result = safe_json_loads(raw)
        assert '\\ b' in result['v']

    def test_valid_unicode_escape_preserved(self):
        raw = r'{"name": "\u0041\u0042\u0043"}'
        result = safe_json_loads(raw)
        assert result['name'] == 'ABC'

    def test_mixed_valid_and_invalid_escapes(self):
        """A string with both valid \\n and invalid \\p in the same value."""
        raw = r'{"v": "line1\nand $\psi$"}'
        # \n is a valid escape, so json.loads won't crash —
        # it succeeds on first try (pass-through).
        result = safe_json_loads(raw)
        assert '\n' in result['v']  # newline character
        # NOTE: \p triggers crash so the fixed path is taken,
        # but we need to check: does it break on first try?
        # Actually \n is valid AND \p is invalid → first json.loads WILL crash.
        # After fix: \n stays as \n (valid), \p→\\p.

    def test_only_valid_escapes_no_fix_applied(self):
        """Strings with only valid escapes should parse on first try."""
        raw = r'{"v": "tab\there\nnewline"}'
        result = safe_json_loads(raw)
        assert result['v'] == 'tab\there\nnewline'


# ── 6. Realistic LLM output patterns ────────────────────────────────


class TestRealisticLLMOutputs:
    """Patterns observed from real LLM entity extraction outputs."""

    def test_extracted_entities_format(self):
        raw = (
            '{"extracted_entities": ['
            '{"name": "Euler\'s formula", "entity_type_id": 0},'
            '{"name": "complex analysis", "entity_type_id": 1}'
            ']}'
        )
        result = safe_json_loads(raw)
        assert len(result['extracted_entities']) == 2

    def test_triplet_extraction(self):
        raw = (
            r'{"triplets": ['
            r'{"subject": "$\vec{F}$", "predicate": "equals", '
            r'"object": "$m \cdot \vec{a}$"}'
            r']}'
        )
        result = safe_json_loads(raw)
        triplet = result['triplets'][0]
        assert '\\vec{F}' in triplet['subject']
        assert '\\cdot' in triplet['object']

    def test_node_summary_with_latex(self):
        raw = (
            r'{"summary": "The Hamiltonian $\hat{H}$ governs time evolution '
            r'via $i\hbar \partial_t |\psi\rangle = \hat{H}|\psi\rangle$."}'
        )
        result = safe_json_loads(raw)
        assert '\\hat{H}' in result['summary']
        assert '\\hbar' in result['summary']

    def test_deeply_nested_json(self):
        raw = (
            r'{"data": {"inner": {"entities": ['
            r'{"name": "$\Delta x \cdot \Delta p \geq \hbar/2$"}'
            r']}}}'
        )
        result = safe_json_loads(raw)
        name = result['data']['inner']['entities'][0]['name']
        assert '\\Delta' in name
        assert '\\hbar' in name

    def test_large_number_of_entities(self):
        """Ensure regex doesn't choke on larger payloads."""
        entities = []
        for i in range(50):
            entities.append(f'{{"name": "entity_{i} $\\\\xi_{i}$", "id": {i}}}')
        raw = '{"entities": [' + ', '.join(entities) + ']}'
        result = safe_json_loads(raw)
        assert len(result['entities']) == 50

    def test_chemical_formulas(self):
        """Chemical notation sometimes uses backslash commands too."""
        raw = r'{"reaction": "$\ce{H2O -> H+ + OH-}$"}'
        result = safe_json_loads(raw)
        assert '\\ce{' in result['reaction']


# ── 7. Idempotency and consistency ───────────────────────────────────


class TestIdempotency:
    """Verify behavior is consistent and idempotent."""

    def test_double_call_same_result(self):
        raw = r'{"name": "$\hat{H}$"}'
        r1 = safe_json_loads(raw)
        # Serialise back and parse again
        r2 = safe_json_loads(json.dumps(r1))
        assert r1 == r2

    def test_valid_json_not_modified(self):
        """For valid JSON, result must exactly match json.loads."""
        cases = [
            '{"a": 1, "b": "hello"}',
            '[1, 2, 3]',
            '"just a string"',
            '42',
            'null',
            '{"nested": {"deep": [true, false, null]}}',
        ]
        for raw in cases:
            assert safe_json_loads(raw) == json.loads(raw)
