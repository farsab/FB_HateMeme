"""
Usage:
  python main.py train    
  python main.py eval     
"""
import sys, torch
from train import get_trainer
from evaluate import evaluate

def train():
    trainer = get_trainer()
    trainer.train()
    trainer.save_model("./ckpt/lora_clip")
    print("✅ Training complete – model saved to ./ckpt/lora_clip")

def eval():
    acc, auroc = evaluate("./ckpt/lora_clip/pytorch_model.bin")
    print(f"✅ Accuracy: {acc:.3f} | AUROC: {auroc:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"train", "eval"}:
        print("Usage: python main.py [train|eval]")
        sys.exit(1)
    torch.manual_seed(42)
    {"train": train, "eval": eval}[sys.argv[1]]()
