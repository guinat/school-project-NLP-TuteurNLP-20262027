"""Adaptation LoRA (partie 5) : encodage des dialogues, boucle d'entraînement, vérifications."""
import json
import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import get_linear_schedule_with_warmup

from nlp_tutor.config import (BATCH_SIZE, EVAL_EVERY, LEARNING_RATE, LORA_ALPHA, LORA_DROPOUT, LORA_RANK,
                              MICRO_BATCH_SIZE, NUM_EPOCHS, PATIENCE, SEED, TARGET_MODULES, WARMUP_SHARE)
from nlp_tutor.prompts import TRAINING_SYSTEM

IGNORED = -100      # étiquette ignorée par l'entropie croisée


def encode(tokenizer, messages, system=TRAINING_SYSTEM, last_answer_only=False):
    """input_ids au format de conversation ; étiquettes ignorées partout sauf sur les réponses de
    l'assistant et leur <|im_end|>, qui apprend au modèle à s'arrêter."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    conversation = [{"role": "system", "content": system}] + list(messages)
    ids = tokenizer.apply_chat_template(conversation)
    labels = [IGNORED] * len(ids)
    positions = [k for k, m in enumerate(conversation) if m["role"] == "assistant"]
    for k in positions[-1:] if last_answer_only else positions:
        prefix = tokenizer.apply_chat_template(conversation[:k], add_generation_prompt=True)
        end = len(tokenizer.apply_chat_template(conversation[:k + 1]))
        assert ids[:len(prefix)] == prefix and ids[end - 2] == end_id
        labels[len(prefix):end - 1] = ids[len(prefix):end - 1]
    return {"input_ids": ids, "labels": labels}


def show_mask(tokenizer, example):
    """Le dialogue en clair, avec les tokens non appris entre ⟦ ⟧."""
    text, start = "", 0
    masked = [label == IGNORED for label in example["labels"]]
    # décodage par segments : un caractère peut tenir sur deux tokens
    for end in range(1, len(masked) + 1):
        if end == len(masked) or masked[end] != masked[start]:
            segment = tokenizer.decode(example["input_ids"][start:end])
            text += f"⟦{segment}⟧" if masked[start] else segment
            start = end
    return text


def batches(examples, size, shuffle=False, seed=0):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    return [[examples[i] for i in order[j:j + size]] for j in range(0, len(order), size)]


def collate(examples, pad_id, device):
    # remplissage à droite pour l'entraînement
    length = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), length), pad_id)
    labels = torch.full((len(examples), length), IGNORED)
    mask = torch.zeros((len(examples), length), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        mask[i, :n] = 1
    return input_ids.to(device), mask.to(device), labels.to(device)


def loss_sum(model, examples, pad_id):
    """Somme des −log p(token correct) sur les tokens de réponse, et nombre de ces tokens.

    L'unembedding n'est appliqué qu'aux positions qui ont une cible : on évite de calculer, et de
    garder pour la rétropropagation, 151 936 logits pour chaque token du prompt."""
    network = model.get_base_model() if isinstance(model, PeftModel) else model
    input_ids, mask, labels = collate(examples, pad_id, network.device)
    hidden = network.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
    targets = labels[:, 1:]                 # le token t+1 est la cible de la position t
    keep = targets != IGNORED
    logits = network.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits, targets[keep], reduction="sum"), int(keep.sum())


@torch.no_grad()
def mean_loss(model, examples, pad_id, batch_size=8):
    model.eval()
    total, n = 0.0, 0
    for batch in batches(examples, batch_size):
        s, k = loss_sum(model, batch, pad_id)
        total, n = total + s.item(), n + k
    return total / n


def add_lora(model):
    config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def fingerprint(model):
    return {name: p.detach().double().sum().item() for name, p in model.named_parameters()}


def reload_adapter(model, adapter_dir):
    set_peft_model_state_dict(model, load_file(adapter_dir / "adapter_model.safetensors"))
    model.eval()


def check_frozen_base(model, fingerprint_before):
    """Nombre de tenseurs de base, tenseurs modifiés et tenseurs qui demandent un gradient."""
    base = {name.replace("base_layer.", ""): p for name, p in model.get_base_model().named_parameters()
            if "lora_" not in name}
    changed = [name for name, p in base.items() if p.detach().double().sum().item() != fingerprint_before[name]]
    return len(base), changed, sum(p.requires_grad for p in base.values())


def train(model, encoded, pad_id, adapter_dir, history_path):
    """Boucle du cours : lot, prédiction, perte, rétropropagation, mise à jour.

    Évalue la validation tous les EVAL_EVERY pas, sauvegarde l'adaptateur à chaque nouveau minimum et
    s'arrête après PATIENCE évaluations sans progrès."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(encoded["train"]) / BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, round(WARMUP_SHARE * total_steps), total_steps)

    history = {"step": [], "train_loss": [], "eval_step": [0],
               "val_loss": [mean_loss(model, encoded["validation"], pad_id)]}
    best, best_step, stale, step = history["val_loss"][0], 0, 0, 0
    print(f"validation loss before training: {best:.3f}")
    print(f"{len(encoded['train'])} examples, {steps_per_epoch} steps per epoch, {total_steps} steps at most\n")

    t0, stop = time.perf_counter(), False
    for epoch in range(NUM_EPOCHS):
        for batch in batches(encoded["train"], BATCH_SIZE, shuffle=True, seed=SEED + epoch):
            model.train()
            n_batch = sum(sum(label != IGNORED for label in e["labels"]) for e in batch)
            batch_loss = 0.0
            for micro_batch in batches(batch, MICRO_BATCH_SIZE):
                s, _ = loss_sum(model, micro_batch, pad_id)
                (s / n_batch).backward()            # moyenne sur tous les tokens du lot
                batch_loss += s.item() / n_batch
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            history["step"].append(step)
            history["train_loss"].append(batch_loss)

            if step % EVAL_EVERY == 0 or step == total_steps:
                val_loss = mean_loss(model, encoded["validation"], pad_id)
                history["eval_step"].append(step)
                history["val_loss"].append(val_loss)
                mark = ""
                if val_loss < best:
                    best, best_step, stale = val_loss, step, 0
                    model.save_pretrained(adapter_dir)
                    mark = "  <- best, saved"
                else:
                    stale += 1
                train_mean = np.mean(history["train_loss"][-EVAL_EVERY:])
                print(f"step {step:3d}/{total_steps} | epoch {epoch + 1} | train loss {train_mean:.3f} | "
                      f"val loss {val_loss:.3f} | {(time.perf_counter() - t0) / 60:4.1f} min{mark}")
                if stale >= PATIENCE:
                    stop = True
                    break
        if stop:
            print(f"\nstopped at step {step}/{total_steps}: no validation progress for {PATIENCE} evaluations")
            break

    history.update(best_step=best_step, best_val_loss=best, duration_min=(time.perf_counter() - t0) / 60,
                   steps_per_epoch=steps_per_epoch)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=1)
    return history


def print_training_log(history):
    """Réaffiche le journal d'un entraînement passé à partir de son historique (TRAIN = False)."""
    best = history["val_loss"][0]
    print(f"validation loss before training: {best:.3f}  (reloaded history)\n")
    for step, val_loss in zip(history["eval_step"][1:], history["val_loss"][1:]):
        i = history["step"].index(step)
        train_mean = np.mean(history["train_loss"][max(0, i + 1 - EVAL_EVERY):i + 1])
        mark = "  <- best" if val_loss < best else ""
        best = min(best, val_loss)
        print(f"step {step:3d} | train loss {train_mean:.3f} | val loss {val_loss:.3f}{mark}")


def plot_loss_curves(history, path=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history["step"], history["train_loss"], color="tab:blue", alpha=0.25, label="train (per batch)")
    smoothed = pd.Series(history["train_loss"]).rolling(EVAL_EVERY, min_periods=1).mean()
    ax.plot(history["step"], smoothed, color="tab:blue", label=f"train (mean over {EVAL_EVERY} batches)")
    ax.plot(history["eval_step"], history["val_loss"], "o-", color="tab:orange", label="validation")
    ax.axvline(history["best_step"], color="gray", ls="--", lw=1, label="best validation loss")
    for e in range(1, NUM_EPOCHS):
        ax.axvline(e * history["steps_per_epoch"], color="black", ls=":", lw=0.8)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("cross-entropy (answer tokens)")
    ax.set_title("LoRA adaptation of Qwen2.5-0.5B-Instruct")
    ax.legend()
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    return fig
