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


def get_subset(
    dataset: VG100KDataset,
    number_upper_bound: int = 5,
    limit: int = 5000
):
    """
    Get a subset of the dataset with only simple questions and answers less than or equal to the number_upper_bound.
    Limits the number of items in the subset.

    Args:
        dataset: Dataset to sample from.
        number_upper_bound: Upper bound for the answer.
        limit: Number of items in the subset.
    Returns:
        Subset of the dataset.
    """
    subset_indices = []
    for i, item in enumerate(dataset):
        if item['answer'] <= number_upper_bound and item['is_simple']:
            subset_indices.append(i)

    torch.manual_seed(42)
    subset_indices = torch.randperm(len(subset_indices)).tolist()
    subset_indices = subset_indices[:limit]    
    return Subset(dataset, subset_indices)


def get_model(model_name: str, device: str):
    """
    Load the model and move it to the device.

    Args:
        model_name: Model name.
        device: Device to move the model to.
    Returns:
        Model.
    """
    model = VLM(model_name)
    model.to(device)
    return model



def inference_dataset(model: VLM, dataset: VG100KDataset) -> list[dict[str, tp.Any]]:
    """
    Base function to perform inference on the dataset.

    Args:
        model: Model to perform inference with.
        dataset: Dataset to perform inference on.
    Returns:
        List of dictionaries with the results.
    """
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
    subset = get_subset(dataset)

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
