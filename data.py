"""JSONL trajectory storage.  JSONL keeps experiments inspectable and append-only."""

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator


def append_transition(path: str, transition: Dict) -> None:
    """Append one serialisable decision record to a data set."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as output:
        output.write(json.dumps(transition, separators=(",", ":")) + "\n")


def load_transitions(path: str) -> Iterator[Dict]:
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                record = json.loads(line)
                if "features" not in record or "action" not in record or "reward" not in record:
                    raise ValueError(f"{path}:{line_no}: expected features, action, reward")
                yield record

