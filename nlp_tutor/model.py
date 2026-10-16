"""Chargement du tokenizer, du modèle de base et de l'adaptateur LoRA."""
import random

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nlp_tutor.config import ADAPTER_DIR, MODEL_NAME, SEED


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"   # génération par lots : tous les prompts finissent au même endroit
    return tokenizer


def load_model(device, name=MODEL_NAME):
    # float32 : sur un CPU sans instructions bfloat16 natives, c'est le format le plus rapide
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device)
    model.eval()
    return model


def load_adapted_model(device, adapter_dir=ADAPTER_DIR):
    tokenizer = load_tokenizer()
    model = PeftModel.from_pretrained(load_model(device), adapter_dir)
    model.eval()
    return tokenizer, model
