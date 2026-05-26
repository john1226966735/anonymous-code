import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Sample:
    id: str
    question: str
    topic_entity: str
    answers: List[str]
    graph: List[List[str]]
    plan_steps: List[str]


class CWQSampleDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.splits = {}
        for split in ["train", "valid", "test"]:
            path = self.root / f"{split}.json"
            with path.open("r", encoding="utf-8") as f:
                records = json.load(f)
            self.splits[split] = [Sample(**record) for record in records]

    def __len__(self):
        return sum(len(v) for v in self.splits.values())

    def split(self, name: str) -> List[Sample]:
        return self.splits[name]

    def all_examples(self) -> List[Sample]:
        return self.splits["train"] + self.splits["valid"] + self.splits["test"]

    @staticmethod
    def entity_names(sample: Sample) -> List[str]:
        names = {sample.topic_entity}
        for h, _, t in sample.graph:
            names.add(h)
            names.add(t)
        names.update(sample.answers)
        return sorted(names)

    @staticmethod
    def relation_names(sample: Sample) -> List[str]:
        rels = {r for _, r, _ in sample.graph}
        return sorted(rels)

    @staticmethod
    def answer_aliases(sample: Sample) -> List[str]:
        return sample.answers

    def summary(self) -> Dict[str, int]:
        return {split: len(samples) for split, samples in self.splits.items()}

