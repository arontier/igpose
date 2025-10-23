import torch
from peft import LoraConfig, get_peft_model, PeftModel

def init_lora(target_modules, lora_r=16, lora_alpha=64, lora_dropout=0.2):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules if not target_modules is None else 'all-linear', # Specific to SAGEConv's Linear layers
        lora_dropout=lora_dropout,
        bias="none",
    )
    return lora_config

def get_lora_model(base_model, target_modules=None, **kargs):
    lora_config = init_lora(target_modules, **kargs)
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model

def load_lora_model(base_model, base_model_path, lora_model_path):
    base_model.load_state_dict(torch.load(base_model_path), strict=False)
    inference_model = PeftModel.from_pretrained(base_model, lora_model_path)
    return inference_model

def save_lora_model(model, lora_model_path):
    model.save_pretrained(lora_model_path)