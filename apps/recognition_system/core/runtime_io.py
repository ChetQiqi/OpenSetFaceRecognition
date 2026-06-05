import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_jsonl(path: str, event: Dict) -> None:
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict], fieldnames: Iterable[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)