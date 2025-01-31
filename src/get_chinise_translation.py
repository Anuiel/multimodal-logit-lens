import json
from copy import deepcopy
from collections import defaultdict
import typing as tp

from torch.utils.data import Subset
from tqdm import tqdm

from dataset import VG100KDataset, format_template
from model import VLM
from utils import is_chinise_str


def get_stratified_subset(
    dataset: VG100KDataset,
    numbers: int = 10,
    number_of_items_per_class: int = 30,
):
    subset_indices = []
    for number in range(numbers):
        items_collected = 0
        for i, item in enumerate(dataset):
            if item['answer'] == number:
                subset_indices.append(i)
                items_collected += 1
                if items_collected == number_of_items_per_class:
                    break
    return Subset(dataset, subset_indices)


def get_model(model_name: str, device: str):
    model = VLM(model_name)
    model.to(device)
    return model


def inference_dataset(model: VLM, dataset: VG100KDataset) -> list[dict[str, tp.Any]]:
    result = []
    for sample in tqdm(dataset):
        image = sample['image']
        text = format_template(sample['question'])
        output = model.inference(image, text)
        output_dict = deepcopy(sample)
        output_dict.update(output)
        result.append(output_dict)
    return result


def main():
    dataset = VG100KDataset(
        json_path='test.json',
        image_dir='VG_100K_2'
    )
    subset = get_stratified_subset(dataset)

    device = 'cuda:0'
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = get_model(model_name, device)

    result = inference_dataset(model, subset)

    number_to_chinise_map: dict[int, set[str]] = defaultdict(set)
    for item in result:
        output_number = item['output_text']
        for layer_number in range(20, 29):
            topk_tokens = item['hidden_states_features'][f"layer_{layer_number}"]['topk_tokens']
            for token in topk_tokens:
                if is_chinise_str(token):
                    number_to_chinise_map[int(output_number)].add(token)
    result: dict[str, list[str]] = {}
    for number, chinise_tokens in number_to_chinise_map.items():
        result[str(number)] = list(chinise_tokens)
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
