import json
from typing import List, Dict, Any


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data
