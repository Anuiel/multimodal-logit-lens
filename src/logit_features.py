import json
from dataclasses import dataclass, field


@dataclass
class LayerFeatures:
    """
    Class for storing the features of a single layer.

    Attributes:
        topk_probs: List of probabilities for the top-k tokens.
        topk_tokens: List of top-k tokens.
    """
    topk_probs: list[float]
    topk_tokens: list[str]


@dataclass
class LogitEntry:
    """
    Class for storing features of a single model pass

    Attributes:
        hidden_features: Dictionary of hidden features for each layer.
        output_number: Output number.
        question: Question.
        answer: Answer.
        is_simple: Whether the question is simple.
        image_path: Path to the image.
    """
    hidden_features: dict[str, LayerFeatures]
    output_number: str
    question: str
    answer: int
    is_simple: bool
    image_path: str


@dataclass
class LogitsData:
    """
    Class for storing the logits data for dataset.

    Attributes:
        entries: List of LogitEntry
    """
    entries: list[LogitEntry] = field(default_factory=list)


def load_logits_json(path: str = "logit.json") -> LogitsData:
    """
    Parse LogitsData from the json file.

    Args:
        path: Path to the json file.
    Returns:
        LogitsData.
    """
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
