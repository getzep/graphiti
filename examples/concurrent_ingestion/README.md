# Concurrent ingestion example

`ingest_jsonl.py` ingests the episodes of a JSONL file into a new graph on embedded FalkorDB. It uses serial
ingestion or concurrent ingestion, and prints the time and the graph counts as JSON. Use it to compare the two
modes on your own data.

## Requirements

- Python 3.12 or later
- `pip install "graphiti-core[falkordblite,tracing]"`
- The environment variable `OPENAI_API_KEY`

## Input

Each line of the file is one JSON object. `content` and `reference_time` are necessary.

```json
{"name": "note-1", "content": "Maria Lopez joined Northwind Robotics in March 2024.", "reference_time": "2024-03-04T09:00:00Z", "source_description": "notes"}
```

The script ingests the lines in file order.

## Run

```bash
# Serial ingestion
python ingest_jsonl.py episodes.jsonl --window 0

# Concurrent ingestion with a window of 4
python ingest_jsonl.py episodes.jsonl --window 4
```

`--limit N` ingests only the first N lines.

## Output

| Field | Content |
|---|---|
| `seconds`, `episodes_per_minute` | The time from the first call to the end of the last commit and the last enrichment. |
| `entity_node_count`, `entity_edge_count` | The size of the graph. |
| `nodes_with_a_shared_name` | The number of entity nodes that have the same normalized name as a different node. |
| `ended_edge_count` | The number of entity edges with an `invalid_at` or an `expired_at` value. |
| `sequence_is_complete` | Concurrent ingestion only. `true` when each episode has a sequence number and no number is missing. |
| `commit_counters` | Concurrent ingestion only. The sum of the repair counts of the `ingest.commit` spans. |

The two modes do not give an equal graph, because the LLM output changes between runs and the LLM sees the
episodes in a different context. Run serial ingestion two times to measure the difference between two runs of the
same mode, and compare the difference between the modes with that value.
