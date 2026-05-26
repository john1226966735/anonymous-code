import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hits1(pred: str, golds: Iterable[str]) -> int:
    p = normalize(pred)
    return int(any(p == normalize(g) for g in golds))


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_predictions(predictions: List[Dict]) -> Dict[str, float]:
    entity_hits = []
    answer_hits = []
    for row in predictions:
        entity_hits.append(int(row["top1"] in row["gold_answers"]))
        answer_hits.append(hits1(row["pred_answer"], row["gold_answers"]))
    return {
        "Entity Hits@1": sum(entity_hits) / max(1, len(entity_hits)),
        "Answer Hits@1": sum(answer_hits) / max(1, len(answer_hits)),
        "N": len(predictions),
    }

