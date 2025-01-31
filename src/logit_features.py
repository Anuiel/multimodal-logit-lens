import json
from dataclasses import dataclass, field


@dataclass
class LayerFeatures:
    topk_probs: list[float]
    topk_tokens: list[str]


@dataclass
class LogitEntry:
    hidden_features: dict[str, LayerFeatures]
    output_number: str
    question: str
    answer: int
    is_simple: bool
    image_path: str


@dataclass
class LogitsData:
    entries: list[LogitEntry] = field(default_factory=list)


def load_logits_json(path: str = "logit.json") -> LogitsData:
    with open(path, 'r') as f:
        data = json.load(f)
        logits_data = LogitsData(
            entries=[
                LogitEntry(
                    hidden_features={k: LayerFeatures(**v) for k, v in entry['hidden_features'].items()},
                    output_number=entry['output_number'],
                    question=entry['question'],
                    answer=entry['answer'],
                    is_simple=entry['is_simple'],
                    image_path=entry['image_path']
                )
            for entry in data
        ])
    return logits_data
