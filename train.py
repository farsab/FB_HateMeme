from transformers import TrainingArguments, Trainer
from data import load_hateful_memes
from model import HateCLIP
import torch

def get_trainer():
    model = HateCLIP()
    ds_train = load_hateful_memes("train[:500]")
    ds_val   = load_hateful_memes("validation[:200]")

    args = TrainingArguments(
        output_dir="./ckpt",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    def collate(batch):
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch]) for k in keys}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=collate,
        compute_metrics=None,
    )
    return trainer
