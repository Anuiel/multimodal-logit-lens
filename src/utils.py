import json
import random

import numpy as np
import torch


def is_chinise_str(s: str) -> bool:
    """
    Function to check if the string contains chinise characters.
    https://stackoverflow.com/questions/34587346/python-check-if-a-string-contains-chinese-character

    Args:
        s: String to check.
    Returns:
        True if the string contains chinise characters, False otherwise.
    """
    return any('\u4e00' <= c <= '\u9fff' for c in s)


def set_seed(seed: int):
    """
    Sets random seed for reproducibility.
    There are still some sources of non-determinism, like cuDNN.

    Args:
        seed: Seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_chinise_mapping(path: str) -> dict[int, list[str]]:
    """
    Loads chinise mapping from the json file.

    Args:
        path: Path to the json file.
    Returns:
        Chinise mapping with folowing structure:
            - key: Numeric value.
            - value: List of chinise characters that represent the key.
    """
    chinise_mapping: dict[int, list[str]] = json.load(open(path))
    return chinise_mapping


def check_answer(answer: str, correct_answer: int, chinise_mapping: dict[int, list[str]]) -> bool:    
    """
    Function to check is a string corresponds to the correct answer.
    For example, if the correct answer is 1, then the following strings are considered correct:
        - "1"
        - "one"
        - " One"
        - "ONE"
        - "一"
        - "一个"
        - "一つ"

    Args:
        answer: Answer to check.
        correct_answer: Correct answer.
        chinise_mapping: Chinise mapping.
    Returns:
        If the answer is correct.
    """
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
