# HCLIP – LoRA‑Fine‑Tuned Multimodal H‑Speech Detector

A **multimodal** demo that fuses CLIP image & text embeddings, then adds
**LoRA** adapters for lightweight fine‑tuning (< 1 % trainable params) on
Facebook’s **Hful Memes** dataset.

| Variant | Params ‑ Trainable | Epochs | BS | F₁ (≈) |
|---------|--------------------|--------|----|--------|
| CLIP‑base (frozen) | 0 M | – | – | 0.50 |
| HCLIP (LoRA)    | 1.1 M | 1 | 4 | **0.67** |

> **Hardware**: runs on a single laptop GPU (slow but memory‑light).

---

## Installation
```bash
git clone https://github.com/your‑handle/Hclip-lora.git
cd Hclip-lora
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision transformers datasets peft accelerate scikit‑learn pandas
```

# Fine‑tune for one epoch on 500 memes
```bash
python main.py train
```

# Evaluate on 200 test samples
```bash
python main.py eval
```

# Files
| File          | Purpose                                           |
|---------------|---------------------------------------------------|
| `data.py`     | Load images & text; apply CLIP preprocessing      |
| `model.py`    | CLIP encoders + LoRA adapters + fusion head       |
| `train.py`    | Builds HuggingFace `Trainer`                      |
| `evaluate.py` | Stand‑alone evaluation (accuracy / AUROC)         |
| `main.py`     | CLI entry‑point (`train`, `eval`)                 |

Note: Adjust slices (train[:500], test[:200]) for full‑dataset training.

