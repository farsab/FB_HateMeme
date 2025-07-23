import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from data import load_hateful_memes
from model import HateCLIP


@torch.no_grad()
def evaluate(model_path: str = None):
    model = HateCLIP()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    ds_test = load_hateful_memes("test[:200]")
    preds, labels = [], []

    for ex in ds_test:
        out = model(
            input_ids=ex["input_ids"].unsqueeze(0),
            attention_mask=ex["attention_mask"].unsqueeze(0),
            pixel_values=ex["pixel_values"].unsqueeze(0),
        )
        preds.append(torch.sigmoid(out["logits"]).item())
        labels.append(ex["labels"].item())

    acc = accuracy_score(labels, [p > 0.5 for p in preds])
    auroc = roc_auc_score(labels, preds)
    return acc, auroc
