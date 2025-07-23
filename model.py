from typing import Dict

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

_LORA_CFG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
)


class HateCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip = get_peft_model(self.clip, _LORA_CFG)  # add LoRA
        self.proj = nn.Linear(self.clip.config.projection_dim * 2, 1)

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        txt_emb = outputs.text_embeds        # [B, D]
        img_emb = outputs.image_embeds       # [B, D]
        fused = torch.cat([txt_emb, img_emb], dim=1)  # [B, 2D]
        logits = self.proj(fused).squeeze(1)          # [B]
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )
        return {"loss": loss, "logits": logits}
