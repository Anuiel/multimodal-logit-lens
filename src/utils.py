import json
import random

import numpy as np
import torch


def is_chinise_str(s: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in s)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_chinise_mapping(path: str) -> dict[int, list[str]]:
    chinise_mapping: dict[int, list[str]] = json.load(open(path))
    return chinise_mapping


def check_answer(answer: str, correct_answer: int, chinise_mapping: dict[int, list[str]]) -> bool:    
    answer = answer.strip().lower()

    numeric_answer: int | None = None
    if is_chinise_str(answer):
        for k, v in chinise_mapping.items():
            if answer in v:
                numeric_answer = k
    elif answer.isdigit():
        numeric_answer = int(answer)
    else:
        numeric_answer = {
            "zero": 0,
            "none": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5
        }.get(answer, None)
    
    return numeric_answer == correct_answer
