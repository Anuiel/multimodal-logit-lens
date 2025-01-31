import json
from copy import deepcopy
from collections import defaultdict
import typing as tp

import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from dataset import VG100KDataset, format_template
from model import VLM
from utils import set_seed


def get_stratified_subset(
    dataset: VG100KDataset,
    number_upper_bound: int = 5,
    limit: int = 5000
):
    subset_indices = []
    for i, item in enumerate(dataset):
        if item['answer'] <= number_upper_bound and item['is_simple']:
            subset_indices.append(i)

    torch.manual_seed(42)
    subset_indices = torch.randperm(len(subset_indices)).tolist()
    subset_indices = subset_indices[:limit]    
    return Subset(dataset, subset_indices)


def get_model(model_name: str, device: str):
    model = VLM(model_name)
    model.to(device)
    return model


def inference_dataset(model: VLM, dataset: VG100KDataset) -> list[dict[str, tp.Any]]:
    result = []
    for sample in tqdm(DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True, collate_fn=lambda x: x[0])):
        if sample['answer'] > 5:
            continue
        image = sample['image']
        text = format_template(sample['question'])
        output = model.inference(image, text)
        output_dict = deepcopy(sample)
        output_dict.update(output)
        result.append(output_dict)
    return result


def main():
    set_seed(42)
    dataset = VG100KDataset(
        json_path='test.json',
        image_dir='VG_100K_2'
    )
    subset = get_stratified_subset(dataset)

    device = 'cuda:0'
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = get_model(model_name, device)

    inference_result = inference_dataset(model, subset)

    result: list[dict[str, tp.Any]] = []
    for item in inference_result:
        output_number = item['output_text']
        local_result: dict[str, dict[str, str | float]] = defaultdict(dict)
        for layer_number in range(29):
            topk_tokens = item['hidden_states_features'][f"layer_{layer_number}"]['topk_tokens']
            topk_probs = item['hidden_states_features'][f"layer_{layer_number}"]['topk_probs']
            local_result[f"layer_{layer_number}"] = {
                "topk_probs": topk_probs,
                "topk_tokens": topk_tokens
            }
        result.append({
            "hidden_features": local_result,
            "output_number": output_number,
            "question": item['question'],
            "answer": item['answer'],
            "is_simple": item['is_simple'],
            'image_path': str(item['image_path']),
        })
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
