import click
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import check_answer, is_chinise_str, get_chinise_mapping
from logit_features import LogitsData, load_logits_json


class MetricsCalculation:
    def __init__(self, logits_data: LogitsData, chinise_mapping: dict[int, list[str]]) -> None:
        self.chinise_mapping = chinise_mapping
        self.logits_data = logits_data

    def check_per_layer_accuracy(self) -> pd.DataFrame:
        per_layer_accuracy: dict[str, float] = {f"layer_{i}": 0 for i in range(29)}
        for entry in self.logits_data.entries:
            for layer, features in entry.hidden_features.items():
                per_layer_accuracy[layer] += check_answer(features.topk_tokens[0], entry.answer, self.chinise_mapping)
        per_layer_accuracy = {layer: correct / len(self.logits_data.entries) for layer, correct in per_layer_accuracy.items()}
        per_layer_accuracy: pd.DataFrame = pd.DataFrame(per_layer_accuracy.items(), columns=["layer", "accuracy"])
        return per_layer_accuracy

    def check_for_chinise_character(self) -> pd.DataFrame:
        per_chinise_character: dict[str, dict[int, int]] = {f"layer_{i}": {k: 0 for k in range(5)} for i in range(29)}
        for entry in self.logits_data.entries:
            for layer, features in entry.hidden_features.items():
                topk: list[bool] = [is_chinise_str(token) for token in features.topk_tokens]
                topk = [any(topk[:i+1]) for i in range(5)]
                for i, has_chinise in enumerate(topk):
                    per_chinise_character[layer][i] += has_chinise
        per_chinise_character = {layer: {topk: has_chinise / len(self.logits_data.entries) for topk, has_chinise in topk_dict.items()} for layer, topk_dict in per_chinise_character.items()}
        per_chinise_character = {(layer, topk, has_chinise) for layer, topk_dict in per_chinise_character.items() for topk, has_chinise in topk_dict.items()}
        per_chinise_character: pd.DataFrame = pd.DataFrame(per_chinise_character, columns=["layer", "topk", "has_chinise"])
        return per_chinise_character

    def check_per_answer_per_layer_accuracy(self) -> pd.DataFrame:
        per_answer_per_layer_correct: dict[tuple[str, int], int] = {(f'layer_{layer}', answer): 0 for layer in range(29) for answer in range(6)}
        answer_count: dict[int, int] = {}
        for entry in self.logits_data.entries:
            answer_count[entry.answer] = answer_count.get(entry.answer, 0) + 1
            for layer, features in entry.hidden_features.items():
                per_answer_per_layer_correct[(layer, entry.answer)] += check_answer(features.topk_tokens[0], entry.answer, self.chinise_mapping)
        per_answer_per_layer_accuracy = {key: correct / answer_count[key[1]] for key, correct in per_answer_per_layer_correct.items()}
        tmp = {(layer, answer, accuracy) for (layer, answer), accuracy in per_answer_per_layer_accuracy.items()}
        per_answer_per_layer_accuracy: pd.DataFrame = pd.DataFrame(tmp, columns=["layer", "answer", "accuracy"])
        return per_answer_per_layer_accuracy


@click.command()
@click.option("--logits-path", default="logit.json", help="Path to the logits json file.")
@click.option("--output-path", default=".", help="Path to save the output images.")
@click.option("--chinise-mapping-path", default="chinise_mapping.json", help="Path to the chinise mapping json file.")
def main(logits_path: str, output_path: str, chinise_mapping_path: str):
    chinise_mapping = get_chinise_mapping(chinise_mapping_path)
    logits_data = load_logits_json(logits_path)
    metrics_calculation = MetricsCalculation(logits_data, chinise_mapping)
    per_layer_accuracy = metrics_calculation.check_per_layer_accuracy()
    per_chinise_character = metrics_calculation.check_for_chinise_character()
    per_answer_per_layer_accuracy = metrics_calculation.check_per_answer_per_layer_accuracy()

    plt.figure(figsize=(20, 10))
    sns.barplot(x="layer", y="accuracy", data=per_layer_accuracy)
    plt.title("Per layer accuracy")
    plt.xticks(rotation=45)
    plt.savefig(f"{output_path}/per_layer_accuracy.png")
    plt.show()

    per_chinise_character = per_chinise_character.pivot(index="topk", columns="layer", values="has_chinise")
    per_chinise_character = per_chinise_character[[f"layer_{i}" for i in range(29)]]
    plt.figure(figsize=(20, 10))
    sns.heatmap(per_chinise_character, annot=True, fmt=".2f")
    plt.title("Per layer topk chinise character")
    plt.savefig(f"{output_path}/per_layer_topk_chinise_character.png")
    plt.show()

    per_answer_per_layer_accuracy = per_answer_per_layer_accuracy.pivot(index="answer", columns="layer", values="accuracy")
    per_answer_per_layer_accuracy = per_answer_per_layer_accuracy[[f"layer_{i}" for i in range(29)]]
    plt.figure(figsize=(20, 10))
    sns.heatmap(per_answer_per_layer_accuracy, annot=True, fmt=".2f")
    plt.title("Per answer per layer accuracy")
    plt.savefig(f"{output_path}/per_answer_per_layer_accuracy.png")
    plt.show()


if __name__ == "__main__":
    main()
