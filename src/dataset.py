import typing as tp
import json
import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class VG100KDataset(Dataset):
    def __init__(self, json_path: str, image_dir: str, transform: tp.Callable[[tp.Any], tp.Any] | None = None):
        """
        Args:
            json_path: Path to the json file with annotations.
            image_dir: Directory with all the images.
            transform: Optional transform to be applied on a sample.
        """
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, tp.Any]:
        """
        Args:
            idx: Index of the sample.
        Returns:
            Sample as a dictionary with the following keys:
                - image_path: Path to the image.
                - image: Image as a PIL.Image.
                - answer: Answer to the question.
                - question: Question.
                - is_simple: Whether the question is simple.
        """
        item = self.data[idx]
        image_path = self.image_dir / Path(item['image']).name
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), resample=Image.BILINEAR)
        answer = item['answer']
        question = item['question']
        is_simple = item['issimple'] 

        if self.transform:
            image = self.transform(image)

        sample = {
            'image_path': image_path,
            'image': image,
            'answer': answer,
            'question': question,
            'is_simple': is_simple
        }

        return sample


def format_template(question: str) -> list[dict[str, tp.Any]]:
    """
    Formats the question with chat template suitable for the Instruct model.

    Args:
        question: Question to format.
    Returns:
        Formatted question.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question} Output single word with exact number."}
            ]
        }
    ]