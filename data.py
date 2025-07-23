"""
data.py
--------
Dataset utilities for loading the Facebook Hateful Memes dataset
and returning (pixel_values, text_inputs, label) tuples compatible
with HuggingFace Trainer.
"""
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

_IMAGE_SIZE = 224
_TOKENIZER_NAME = "openai/clip-vit-base-patch32"
_TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

_IMG_TFORM = transforms.Compose(
    [
        transforms.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ]
)


def _preprocess(example: Dict) -> Dict:
    img_path = Path(example["img"])
    image = Image.open(img_path).convert("RGB")
    example["pixel_values"] = _IMG_TFORM(image)
    txt = example["text"]
    enc = _TOKENIZER(txt, truncation=True, padding="max_length", max_length=64)
    example["input_ids"] = torch.tensor(enc["input_ids"])
    example["attention_mask"] = torch.tensor(enc["attention_mask"])
    example["labels"] = torch.tensor(example["label"])
    return example


def load_hateful_memes(split: str = "train[:500]"):
    """
    Returns a processed Dataset with
      - pixel_values  : Tensor(C, H, W)
      - input_ids     : Tensor(seq)
      - attention_mask: Tensor(seq)
      - labels        : Tensor(1)
    """
    ds = load_dataset("hateful_memes", split=split, cache_dir="./cache")
    return ds.map(_preprocess, remove_columns=ds.column_names)
