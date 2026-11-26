"""The YAML settings source reads config.yaml as UTF-8, whatever the locale encoding is."""

import json
import os
import subprocess
import sys
from pathlib import Path

SRC_PATH = Path(__file__).parent.parent / 'src'

LOAD_SCRIPT = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from pydantic_settings import BaseSettings
from config.schema import YamlSettingsSource

class Settings(BaseSettings):
    pass

print(json.dumps(YamlSettingsSource(Settings, Path(sys.argv[2]))()))
"""


def test_yaml_source_reads_utf8_under_a_non_utf8_locale(tmp_path):
    config_path = tmp_path / 'config.yaml'
    description = 'Caf\u00e9 \u2014 \u65e5\u672c\u8a9e \u00ab\u4eba\u00bb'
    config_path.write_bytes(
        (
            'graphiti:\n  entity_types:\n    - name: Person\n      description: '
            + description
            + '\n'
        ).encode('utf-8')
    )

    env = {key: value for key, value in os.environ.items() if key != 'PYTHONUTF8'}
    # Python must not fall back to UTF-8 on its own, so the locale encoding decides.
    env.update(PYTHONUTF8='0', PYTHONCOERCECLOCALE='0', LC_ALL='C', LANG='C')

    result = subprocess.run(
        [sys.executable, '-c', LOAD_SCRIPT, str(SRC_PATH), str(config_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert loaded['graphiti']['entity_types'][0]['description'] == description
